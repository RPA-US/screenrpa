



def quantity_ui_elements_fe_technique(feature_extraction_technique_name, overwrite_info): #enriched_log_output_path="resources/enriched_log_feature_extracted.csv"
    """
    Since not all images have all classes, a dataset with different columns depending on the images will be generated.
    It will depend whether GUI components of every kind appears o only a subset of these. That is why we initiañize a 
    dataframe will all possible columns, and include row by row the results obtained from the predictions

    :param feature_extraction_technique_name: Feature extraction technique name
    :type feature_extraction_technique_name: str
    :param enriched_log_output_path: Path to save the enriched log
    :type enriched_log_output_path: str
    :param overwrite_info: Rewrite log
    :type overwrite_info: bool

    """
    print("TODO")
    log = pd.read_csv(param_log_path, sep=",")
    # screenshot_filenames = [ x + ".npy" for x in log.loc[:,"Screenshot"].values.tolist()]
    screenshot_filenames = log.loc[:, screenshot_colname].values.tolist()

    df = pd.DataFrame([], columns=default_ui_elements_classification_classes)

    for i in range(0, len(images_names)):
        row1 = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        # Acess the previously stored frequencies
        df1 = crop_imgs[images_names[i]]["result_freq_df"]
        if len(df1.columns.tolist()) > 0:
            for x in df1.columns.tolist():
                uiui = nombre_clases.index(x)
                row1[uiui] = df1[x][0]
                df.loc[i] = row1

    """
    Once the dataset corresponding to the ammount of elements of each class contained in each of the images is obtained,
    we merge it with the complete log, adding the extracted characteristics from the images
    """

    log_enriched = log.join(df).fillna(method='ffill')

    """
    Finally we obtain an entiched log, which is turned as proof of concept of our hypothesis based on the premise that if
    we not only capture keyboard or mouse events on the monitorization through a keylogger, but also screencaptures,
    we can extract much more useful information, being able to improve the process mining over said log.
    As a pending task, we need to validate this hypothesis through a results comparison against the non-enriched log.
    We expect to continue this project in later stages of the master
    """
    log_enriched.to_csv(enriched_log_output_path)
    print("\n\n=========== ENRICHED LOG GENERATED: path=" +
          enriched_log_output_path)



def location_ui_elements_fe_technique(feature_extraction_technique_name, overwrite_info):
    print("TODO") # TODO: 

def location_ui_elements_and_plaintext_fe_technique(feature_extraction_technique_name, overwrite_info):
    print("TODO") # TODO: 


