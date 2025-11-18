import os
import json
import re
import numpy as np
from shapely.geometry import Polygon, Point
from tqdm import tqdm
import pandas as pd
import polars as pl
from shapely import STRtree, intersection_all
from shapely import points as from_shapely_points
from apps.analyzer.models import Execution
from apps.featureextraction.utils import read_ui_log_as_dataframe
from concurrent.futures import ThreadPoolExecutor


def filter_intersection_relevant(path_scenario, execution, compos_nparray, rows):
    # First we read and combine the relevant components of all screenshots of the same activity
    from apps.featureextraction.relevantinfoselection.postfilters import (
        filter_relevant_ui_components,
    )

    fixation_regions_path = os.path.join(
        path_scenario + "_results", "postfilter_attention_maps"
    )
    fixation_regions_npys = [
        f
        for f in os.listdir(fixation_regions_path)
        if os.path.isfile(os.path.join(fixation_regions_path, f)) and f.endswith(".npy")
    ]
    activity_screenshots = rows[
        execution.case_study.special_colnames["Screenshot"]
    ].to_list()
    activity_relevant_regions = np.concatenate(
        [
            np.load(os.path.join(fixation_regions_path, f), allow_pickle=True)
            for f in fixation_regions_npys
            if f[:-4] in [os.path.basename(s[:-4]) for s in activity_screenshots]
        ]
    ).ravel()

    # compute the intersection of all relevant components
    regions_intersection = intersection_all(activity_relevant_regions)

    compos_nparray = filter_relevant_ui_components(
        False,
        activity_screenshots,  # hack
        activity_screenshots[0],  # hack, sorry
        compos_nparray,
        regions_intersection,
    )

    return compos_nparray


def combine_ui_element_centroid_aux(
    ui_log_path,
    path_scenario,
    execution,
    use_text,
    gaze_conciliation,
    only_gaze_conciliation=False,
):
    """
    Combine the information of the UI elements and the centroids of the UI elements in the same dataset
    for the same activities
    """
    # Iterate over the images again to find for each centroid the smallest object containing it
    execution_root = path_scenario + "_results"
    metadata_json_root = os.path.join(execution_root, "components_json")
    screenshot_colname = execution.case_study.special_colnames["Screenshot"]
    text_classname = execution.ui_elements_classification.model.text_classname

    if not os.path.exists(
        os.path.join(path_scenario + "_results", "log_enriched.csv")
    ):  # to be applied only to aggregated features
        log = read_ui_log_as_dataframe(ui_log_path, lib="polars")
        log.write_csv(os.path.join(path_scenario + "_results", "log_enriched.csv"))
        del log
    if not os.path.exists(os.path.join(path_scenario + "_results", "pipeline_log.csv")):
        fe_log = read_ui_log_as_dataframe(
            os.path.join(path_scenario + "_results", "log_enriched.csv"), lib="polars"
        )
        pd_log = read_ui_log_as_dataframe(
            os.path.join(path_scenario + "_results", "pd_log.csv"), lib="polars"
        )
        cols_to_drop = pd_log.columns
        cols_to_drop.remove(execution.case_study.special_colnames["Screenshot"])
        fe_log = fe_log.drop(cols_to_drop, strict=False)
        log = pd_log.join(
            fe_log,
            how="inner",
            on=execution.case_study.special_colnames["Screenshot"],
        ).with_row_count("orig_idx")
        del fe_log
        del pd_log
    else:
        log = read_ui_log_as_dataframe(
            os.path.join(path_scenario + "_results", "pipeline_log.csv"), lib="polars"
        ).with_row_count("orig_idx")
    activities = list(
        set(log[execution.case_study.special_colnames["Activity"]].to_list())
    )

    centroid_regex = re.compile(r".*_(\d*\.?\d+)-(\d*\.?\d+)")
    centroid_columns = [
        col
        for col in log.columns
        if centroid_regex.match(col) and log.get_column(col).count() > 0
    ]
    for activity in activities:
        rows = log.filter(
            pl.col(execution.case_study.special_colnames["Activity"]) == activity
        )
        for row in tqdm(
            rows.iter_rows(named=True),
            desc="Updating centroids with classes for each screenshot",
        ):
            screenshot_filename = os.path.basename(row[screenshot_colname])

            # Check if the file exists, if exists, then we can continue
            if os.path.exists(
                os.path.join(metadata_json_root, screenshot_filename + ".json")
            ):
                with open(
                    os.path.join(metadata_json_root, screenshot_filename + ".json"), "r"
                ) as f:
                    data = json.load(f)

                compos_nparray = np.array(data["compos"])

                if gaze_conciliation == "intersection":
                    # Filter only relevant components in all instances of the same activity
                    compos_nparray = filter_intersection_relevant(
                        path_scenario, execution, compos_nparray, rows
                    )

                data["compos"] = list(compos_nparray)
                with open(
                    os.path.join(metadata_json_root, screenshot_filename + ".json"), "w"
                ) as f:
                    json.dump(data, f, indent=4)

                log = centroid_conciliation(
                    log,
                    activity,
                    centroid_columns,
                    row["orig_idx"],
                    data,
                    compos_nparray,
                    use_text,
                    only_gaze_conciliation,
                    text_classname,
                )

    # Copy trace_id column because it gets deleted sometimes
    trace = log[execution.case_study.special_colnames["Case"]]
    variant = log[execution.case_study.special_colnames["Variant"]]
    # Remove columns with the same values
    log = pl.from_pandas(log.to_pandas().T.drop_duplicates().T).drop("orig_idx")
    log = log.with_columns(
        [
            pl.Series(execution.case_study.special_colnames["Case"], trace),
            pl.Series(execution.case_study.special_colnames["Variant"], variant),
        ]
    )
    # Remove nan columns
    # log = log.select(pl.all().fill_nan(None))  # polars does not have dropna for columns
    log = log[[s.name for s in log if not (s.null_count() == log.height)]]
    # Save the updated log
    log.write_csv(os.path.join(execution_root, "pipeline_log.csv"), separator=",")

    # Save relevant uicompo data. This is relevant for aggregated features
    for compo in data["compos"]:
        if compo["id"] in list(map(lambda x: x["id"], compos_nparray)):
            compo["relevant"] = True
        else:
            compo["relevant"] = False

    with open(
        os.path.join(metadata_json_root, screenshot_filename + ".json"), "w"
    ) as f:
        json.dump(data, f, indent=4)

    return 0, 0, 0, 0


def centroid_conciliation(
    log,
    activity,
    centroid_columns,
    index,
    data,
    compos_nparray,
    use_text,
    only_gaze_conciliation,
    text_classname,
):
    """
    Process centroids and match them with UI elements.

    Args:
        rows: DataFrame with the rows for the current activity
        log: Full log dataframe that will be updated
        index: Current row index being processed
        data: UI component data from JSON
        compos_nparray: Array of component data
        use_text: Whether to use text content instead of class
        only_gaze_conciliation: Whether to only remove non-relevant components
        text_classname: Classname used for text elements
    """
    centroid_regex = re.compile(r".*_(\d*\.?\d+)-(\d*\.?\d+)")
    pending = dict()
    # keep legacy-only branch untouched except for parsing optimizations
    if only_gaze_conciliation:
        non_relevant_compos_ids = set(map(lambda x: x["id"], data["compos"])) - set(
            map(lambda x: x["id"], compos_nparray)
        )
        non_relevant_compos = [
            compo for compo in data["compos"] if compo["id"] in non_relevant_compos_ids
        ]
        for compo in non_relevant_compos:
            # Single columns
            if (
                c := f"rpa-us_{compo['centroid'][0]}-{compo['centroid'][1]}"
                in centroid_columns
            ):
                pending[c] = None
            if (
                c
                := f"rpa-us_{compo['centroid'][0]}-{compo['centroid'][1]}_{compo['class']}"
                in centroid_columns
            ):
                pending[c] = None
            c = f"rpa-us_{compo['centroid'][0]}-{compo['centroid'][1]}_{compo.get('text')}"
            if use_text and compo["class"] == text_classname and c in centroid_columns:
                pending[c] = None

            # FIXME: Aggregated columns. How do we handle these? We cannot set them to NaN, and we cannot subtract from it
            # if c := f"numeric__rpa_us_{compo['class']}_{activity}" in log.columns:
            #     pending[c] = np.nan
            # c = f"numeric__rpa_us_{compo.get('text')}_{activity}"
            # if use_text and compo["class"] == text_classname and c in log.columns:
            #     pending[c] = np.nan
    else:
        # Build polygons and keep mapping to compos
        compos_list = list(compos_nparray)
        polygons = []
        poly_to_compo = {}
        for compo in compos_list:
            poly = Polygon(compo["points"])
            polygons.append(poly)
            poly_to_compo[id(poly)] = compo

        # Build spatial index
        if len(polygons) == 0:
            return log

        tree = STRtree(polygons)

        pending = assign_centroids_to_compo(
            centroid_columns,
            centroid_regex,
            tree,
            poly_to_compo,
            use_text,
            text_classname,
        )

    # apply pending assignments in a single vectorized write where possible
    if pending:
        exprs = [
            (
                pl.when(pl.col("orig_idx") == index)
                .then(pl.lit(pending[c]))
                .otherwise(pl.col(c))
                .alias(c)
            )
            if c in pending
            else pl.col(c)
            for c in log.columns
        ]
        return log.with_columns(exprs)

    return log


def assign_centroids_to_compo(
    centroid_columns: list[str],
    centroid_regex,
    tree: STRtree,
    poly_to_compo: dict[int, dict],
    use_text: bool,
    text_classname: str,
) -> dict[str, str]:
    coords = []
    col_indices = []
    # Collect points in bulk
    for idx, col in enumerate(centroid_columns):
        m = centroid_regex.match(col)
        if not m:
            continue
        cx = float(m.groups()[0])
        cy = float(m.groups()[1])
        coords.append((cx, cy))
        col_indices.append(idx)
    if not coords:
        return {}

    coords_arr = np.array(coords)  # shape (M, 2)
    # Create geometry array of points
    points = from_shapely_points(coords_arr[:, 0], coords_arr[:, 1])  # vectorized

    # Bulk query: get all (point_i, poly_j) pairs
    pairs = tree.query(points, predicate="within")
    if len(pairs) == 0:
        return {}

    df = pd.DataFrame(pairs.T, columns=["point_idx", "poly_idx"])
    df["area"] = df["poly_idx"].apply(
        lambda j: tree.geometries[j].area
    )  # point | intersecting polygon | polygon area

    # For each point_idx, pick the poly_idx with minimum area
    idx_min_area = df.groupby("point_idx")["area"].idxmin()
    df_min = df.loc[idx_min_area]  # point | polygon with min area | area

    pending: dict[str, str] = {}
    # For each assigned centroid, choose the right class/text

    # Partition pairs into balanced chunks
    pairs = list(zip(df_min["point_idx"], df_min["poly_idx"]))
    n_threads = min(8, os.cpu_count() or 1)
    chunk_size = max(1, len(pairs) // n_threads)
    chunks = [pairs[i : i + chunk_size] for i in range(0, len(pairs), chunk_size)]

    n_cols = len(centroid_columns)

    # Preallocate object array for results
    pending_arr = np.empty(n_cols, dtype=object)

    # Worker function operating on a chunk of (pt_i, poly_i) pairs
    def worker(rows):
        for pt_i, poly_i in rows:
            poly = tree.geometries[int(poly_i)]
            compo = poly_to_compo.get(id(poly))
            if compo is None:
                continue
            idx = col_indices[int(pt_i)]  # numeric index into centroid_columns
            pending_arr[idx] = (
                compo.get("text")
                if use_text and compo.get("class") == text_classname
                else compo.get("class")
            )

    # Execute threads in parallel
    with ThreadPoolExecutor(max_workers=n_threads) as ex:
        ex.map(worker, chunks)

    # Convert non-empty array entries back into a dictionary
    pending = {
        centroid_columns[i]: val for i, val in enumerate(pending_arr) if val is not None
    }

    return pending


def combine_ui_element_centroid(
    ui_log_path, path_scenario, execution: Execution, gaze_conciliation
):
    return combine_ui_element_centroid_aux(
        ui_log_path, path_scenario, execution, False, gaze_conciliation
    )


def combine_ui_element_centroid_and_text(
    ui_log_path, path_scenario, execution: Execution, gaze_conciliation
):
    return combine_ui_element_centroid_aux(
        ui_log_path, path_scenario, execution, True, gaze_conciliation
    )


def only_gaze_conciliation(
    ui_log_path, path_scenario, execution: Execution, gaze_conciliation
):
    return combine_ui_element_centroid_aux(
        ui_log_path,
        path_scenario,
        execution,
        True,
        gaze_conciliation,
        only_gaze_conciliation=True,
    )
