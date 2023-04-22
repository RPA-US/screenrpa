To use this functionality we have to add the "monitoring" phase to the request as shown below:

```
"phases_to_execute": {
        "monitoring": {
            "user": "1",
            "type": "imotions",
            "configurations": {
                "format": "mht_csv",
                "formatted_log_name": "log",
                "org:resource": "User1",
                "mht_log_filename": "Recording_20230404_1459.mht",
                "eyetracking_log_filename": "ET_RExtAPI-GazeAnalysis.csv",
                "native_slide_events": "Native_SlideEvents.csv",
                "ui_log_adjustment": "0.",
                "gaze_log_adjustment": "0.",
                "separator": ","
            }
        },
        ...
```

It will be necessary to have a file structure similar to the following:
```
experiment_folder/
    Recording_20230404_1459.mht
    ET_RExtAPI-GazeAnalysis.csv
    Native_SlideEvents.csv
    screenshot0001.jpeg
    screenshot0002.jpeg
    screenshot0003.jpeg
    screenshot0004.jpeg
    screenshot0005.jpeg
    ...
```