import os
import json
import re
import numpy as np
from shapely.geometry import Polygon, Point
from tqdm import tqdm
import pandas as pd
from shapely import intersection_all
from apps.analyzer.models import Execution
from apps.featureextraction.utils import read_ui_log_as_dataframe


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
    ].values.tolist()
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

    if not os.path.exists(os.path.join(path_scenario + "_results", "pipeline_log.csv")):
        fe_log = read_ui_log_as_dataframe(
            os.path.join(path_scenario + "_results", "log_enriched.csv")
        )
        pd_log = read_ui_log_as_dataframe(
            os.path.join(path_scenario + "_results", "pd_log.csv")
        )
        cols_to_drop = pd_log.columns.tolist()
        cols_to_drop.remove(execution.case_study.special_colnames["Screenshot"])
        fe_log = fe_log.drop(columns=cols_to_drop, errors="ignore")
        log = pd.merge(
            pd_log,
            fe_log,
            how="inner",
            on=execution.case_study.special_colnames["Screenshot"],
        )
        del fe_log
        del pd_log
    else:
        log = read_ui_log_as_dataframe(
            os.path.join(path_scenario + "_results", "pipeline_log.csv")
        )
    activities = list(
        set(
            log.loc[
                :, execution.case_study.special_colnames["Activity"]
            ].values.tolist()
        )
    )

    for activity in activities:
        rows = log[log[execution.case_study.special_colnames["Activity"]] == activity]
        for index, row in tqdm(
            rows.iterrows(), desc="Updating centroids with classes for each screenshot"
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

                # Both components and centroids as numpy arrays to make it more performant
                compos_nparray = np.array(data["compos"])

                if gaze_conciliation == "intersection":
                    # Filter only relevant components in all instances of the same activity
                    compos_nparray = filter_intersection_relevant(
                        path_scenario, execution, compos_nparray, rows
                    )

                log = centroid_conciliation(
                    rows,
                    log,
                    index,
                    data,
                    compos_nparray,
                    use_text,
                    only_gaze_conciliation,
                    execution.ui_elements_classification.model.text_classname,
                )

    # Copy trace_id column because it gets deleted sometimes
    trace = log[execution.case_study.special_colnames["Case"]]
    variant = log[execution.case_study.special_colnames["Variant"]]
    # Remove columns with the same values
    log = log.T.drop_duplicates().T
    log[execution.case_study.special_colnames["Case"]] = trace
    log[execution.case_study.special_colnames["Variant"]] = variant
    # Remove nan columns
    log = log.dropna(axis=1, how="all")
    # Save the updated log
    log.to_csv(os.path.join(execution_root, "pipeline_log.csv"), index=False)

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
    rows,
    log,
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
    # identifier_-centroidY
    centroid_regex = re.compile(r".*_(\d*\.?\d+)-(\d*\.?\d+)")
    # Get all the columns that match the regex and do not contain only nan values
    centroid_columns = [
        col
        for col in rows.columns
        if centroid_regex.match(col) and not rows[col].isnull().all()
    ]

    if only_gaze_conciliation:
        non_relevant_compos_ids = set(map(lambda x: x["id"], data["compos"])) - set(
            map(lambda x: x["id"], compos_nparray)
        )
        non_relevant_compos = [
            compo for compo in data["compos"] if compo["id"] in non_relevant_compos_ids
        ]
        for compo in non_relevant_compos:
            for col in centroid_columns:
                centroid = np.array(
                    [
                        centroid_regex.match(col).groups()[0],
                        centroid_regex.match(col).groups()[1],
                    ]
                )
                classname = col.split("_", maxsplit=2)[2]

                if (
                    compo["centroid"][0] == centroid[0]
                    and compo["centroid"][1] == centroid[1]
                ):
                    if (
                        use_text
                        and compo["class"] == text_classname
                        and compo["text"] == classname
                    ):
                        log.at[:, col] = np.nan
                    elif compo["class"] == classname:
                        log.at[:, col] = np.nan
    else:
        # Pre-compute Polygon objects to avoid creating them in each iteration
        compos_polygons = [
            (Polygon(compo["points"]), compo) for compo in compos_nparray
        ]

        # Match each centroid with the smallest object containing it using Polygon from shapely
        for col in centroid_columns:
            centroid = np.array(
                [
                    centroid_regex.match(col).groups()[0],
                    centroid_regex.match(col).groups()[1],
                ]
            )
            centroid_point = Point(centroid.astype(float))
            containing_compos = [
                (compo, poly.area)
                for poly, compo in compos_polygons
                if poly.contains(centroid_point)
            ]
            if len(containing_compos) == 0:
                continue
            compo = min(containing_compos, key=lambda x: x[1])[0]

            # Insert the class of the smallest object containing the centroid
            if use_text and compo["class"] == text_classname:
                log.at[index, col] = compo["text"]
            else:
                log.at[index, col] = compo["class"]

    return log


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
