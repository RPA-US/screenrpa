# Deployment instruction of RIM for a local development environment

## Create .env for Docker container

Using the `.env.sample` file available as a template, set the values for the user, password and database you want to use with PostgreSQL for this project.

## Create docker container

Open a terminal emulator, navigate to the root directory of the project and run **`docker-compose up`** from  to create the container. This container will be composed of two images:
- Rim-dev: Image with the project code and dependencies, mounted in /rim
- db: PostgreSQL image to store the database

## Open the project

To access the contents of the container, you can use Visual Studio Code or any other IDE with support for it.

For Visual Studio Code, install the [Docker](https://marketplace.visualstudio.com/items?itemName=ms-azuretools.vscode-docker) extension, navigate to its tab, and inside the container you just created you will see the diferent images that compose it. Right click on the rim image and select Attach to Visual Studio Code.

Once a new window has popped up, wait until the container is fully loaded, then click on open directory and select `/rim` as the target.

Finally, install the [Python](https://marketplace.visualstudio.com/items?itemName=ms-python.python) extension inside the container.

## Configuration DB
Firstly, you need configure the Database for the project. To do this, create an *.env* file in the folder *rim* with the following contents:
```
DB_NAME=<database_name>
DB_HOST=db
DB_PORT=5432
DB_USER=<user>
DB_PASSWORD=<password>
DJANGO_SETTINGS_MODULE=rim.settings
METADATA_PATH=/rim/resources/input/metadata
API_VERSION=api/<version>/
GUI_COMPONENTS_DETECTION_CROPPING_THRESHOLD=2
GAZE_MINIMUM_TIME_STARING=10
RESULTS_TIMES_FORMAT=seconds
DECISION_TREE_TRAINING_FOLDERNAME=decision-tree
GUI_QUANTITY_DIFFERENCE=1
```

## Project initialization

In the project directory, open a terminal and run:

**`source ./venv/bin/activate`**

Activate the python virtual environment in which we have all our project dependencies installed.

**`python manage.py makemigrations`**

To create a DB model.

**`python manage.py migrate`**

To create the tables in the DB based on project models.

**`python manage.py loaddata configuration/db_populate.json`**

To insert initial data in DB.

**`python manage.py runserver`**

Runs the app in the debug mode. If you want to init in deploy mode, change in the *rim/settings.py* file, the *DEBUG* mode attribute to False.

**`redis-server`**

Initialize the redis server as your celery broker.

**`python -m celery -A rim worker --concurrency 1`**

Starts the celery worker for the rim application, with 1 being the number of celery tasks that can be executed simultaneously.

Celery, on pair with redis, is used on this project to isolate the execution of time and resource intensive tasks in different virtual threads, give the ability to set up a queue for them and limit the amount of simultaneous resource intensive processes executed.

## Learn More

You can learn more about the deploy of the aplication backend in the [Django documentation](https://docs.djangoproject.com/en/4.0/).

You can learn more about distributed tasks queues in the [Celery documentation](https://docs.celeryq.dev/en/stable/)

## Configure git repository

In the case that the git repository has not been properly configured by default, run the following command while on the project root directory:

**`git checkout -t origin/<branch_name>`**

## Run a test experiment
We will run a test experiment to make sure everything was set up properly. To run an experiment you will need to copy the rim-resources folder into the the `/rim` folder in the container, and change the name to `resources`.

Move the `.env` inside `rim-resources` to the `rim` directory inside the project and replace the placeholders with the user and password you chose for the database.

Execute the commands of the project initialization section to run the project.

Open Postman or any other software that lets you send requests via http.

Create a POST request to http://127.0.0.1:8000/api/v1/analyzer/ and copy this json into the body of the request:
```json
{
    "title": "Test Case Study",
    "exp_foldername": "Basic_exp",
    "phases_to_execute": {
        "ui_elements_detection": {
            "eyetracking_log_filename": "eyetracking_log.csv",
            "add_words_columns": false,
            "overwrite_info": true,
            "algorithm": "legacy"
        },
        "ui_elements_classification": {
            "model_weights": "resources/models/custom-v2.h5",
            "model_properties": "resources/models/custom-v2-properties.json",
            "overwrite_info": true,
            "ui_elements_classification_classes": ["button",
            "checkbox_checked",
            "checkbox_unchecked",
            "image",
            "radio",
            "scroll",
            "seekbar",
            "text",
            "text_input",
            "toggle_switch"],
            "classifier": "uied"
        },
        "extract_training_dataset": {
            "columns_to_ignore": ["Coor_X", "Coor_Y"]
        }
    },
    "special_colnames": {
        "Case": "Case",
        "Activity": "Activity",
        "Screenshot": "Screenshot", 
        "Variant": "Variant",
        "Timestamp": "Timestamp",
        "eyetracking_recording_timestamp": "Recording timestamp",
        "eyetracking_gaze_point_x": "Gaze point X",
        "eyetracking_gaze_point_y": "Gaze point Y"
    },
    "decision_point_activity": "B",
    "exp_folder_complete_path": "/rim/resources/input/Basic_exp copy/",
    "gui_class_success_regex": "CheckBox_B or ImageView_B or TextView_B",
    "gui_quantity_difference": 1,
    "scenarios_to_study": null,
    "drop": null
}
```
This request will take some time to execute, but if everything was done correctly, you will recieve the following message:
```json
{
    "message": "Case Study generated"
}
``` 