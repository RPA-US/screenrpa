import pandas as pd


# Auxiliar function of "extract_training_dataset"
def flat_dataset_row(data, 
                     columns,
                     param_timestamp_column_name,
                     param_variant_column_name,
                     columns_to_drop,
                     param_decision_point_activity,
                     actions_columns):
    """
    With this function we convert the log into a dataset, such that we flatten all the existing registers over the same case,
    resulting on a single row per case. Fot this flattening we only take in account those registers relative to the activities previous
    to the one indicated in para_decision_point_activity, including this last one. The names of the activities colums are concatinated
    with their id, for example, timestamp_A, timestamp_B, etc.

    :param data: Dict which keys correspond to the identification of each case and which values are the activies asociated to said case, along with their information
    :type data: dict
    :param columns: Names of the dataset colums wanted to be stored for each activity
    :type columns: list
    :param param_timestamp_column_name: nombre de la columna donde se almacena el timestamp
    :type param_timestamp_column_name: str
    :param param_variant_column_name: nombre de la columna donde se almacena la variante, que será la etiqueta de nuestro problema
    :type param_variant_column_name: str
    :param columns_to_drop: Names of the colums to remove from the dataset
    :type columns_to_drop: list
    :param param_decision_point_activity: Identificatior of the activity inmediatly previous to the decision point which "why" wants to be discovered
    :type param_decision_point_activity: str
    :param actions_columns: 
    :type actions_columns: list
    :returns: Dataset
    :rtype: DataFrame
    """
    add_case = False  # Set True to add case column
    df_content = []
    new_list = [col_name for col_name in columns if col_name not in actions_columns]
    for case in data["cases"]:
        # print(case)
        timestamp_start = data["cases"][case]["A"].get(
            key=param_timestamp_column_name)
        timestamp_end = data["cases"][case][param_decision_point_activity].get(
            param_timestamp_column_name)
        variant = data["cases"][case]["A"].get(key=param_variant_column_name)
        if add_case:
            row = [variant, case, timestamp_start,
                   timestamp_end]  # ADDING CASE COLUMN
            headers = ["Variant", "Case", "Timestamp_start", "Timestamp_end"]
        else:
            # WITHOUT CASE COLUMN
            row = [variant, timestamp_start, timestamp_end]
            headers = ["Variant", "Timestamp_start", "Timestamp_end"]
        for act in data["cases"][case]:
            # case
            # variant_id
            if act != param_decision_point_activity:
                # Add sufix with the corresponding activity letter to all columns
                row.extend(data["cases"][case][act].drop(
                    columns_to_drop).values.tolist())
                for c in columns:
                    headers.append(c+"_"+act)
            else:
                row.extend(data["cases"][case][act].drop(
                    columns_to_drop).drop(actions_columns).values.tolist())
                for c in new_list:
                    headers.append(c+"_"+act)
                break
        # Introduce the row with the info of all activities of the case in the dataset
        df_content.append(row)
    df = pd.DataFrame(df_content, columns=headers)
    return df
