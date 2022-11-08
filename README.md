# RIM tool
Relevance Information Mining tool

## Before run
For development in Windows, make sure you have [Docker](https://docs.docker.com/desktop/install/windows-install/) installed and working. On Linux this is optional, since all Python packages on this project are compatible with it.

If you are going to run this on your local machine (not a container), you need to have [Python](https://www.python.org/downloads/) and [PostgreSQL](https://www.postgresql.org/download/) installed.

If desired, you can create an isolated installation of the project requirements by creating a [virtual environment](https://docs.python.org/3/library/venv.html#:~:text=A%20virtual%20environment%20is%20a,part%20of%20your%20operating%20system.).

## Create .env for Docker container

Using the `.env.sample` file available as a template, set the values for the user, password and database you want to use with PostgreSQL for this project.

## Create docker container

Open a terminal emulator, navigate to the root directory of the project and run **`docker-compose up`** from  to create the container. This container will be composed of two images:
- Rim-dev: Image with the project code and dependencies, mounted in /rim
- db: PostgreSQL image to store the database

## Open the project

To access the contents of the container, you can use Visual Studio Code or any other IDE with support for it.

For Visual Studio Code, install the [Docker]() extension, navigate to its tab, and inside the container you just created you will see the diferent images that compose it. Right click on the rim image and select Attach to Visual Studio Code.

Once a new window has popped up, wait until the container is fully loaded, then click on open directory and select `/rim` as the target.

## Configuration DB
Firstly, you need configure the Database for the project. To do this, create an *.env* file in the folder *rim* with the following contents:
```
-  DB_NAME="Database name"
-  DB_HOST="Database URL"
-  DB_PORT="Database access port"
-  DB_USER="Database user to access. Use a new user with limited credentials"
-  DB_PASSWORD="Password for the previous user"
-  DJANGO_SETTINGS_MODULE=rim.settings
-  METADATA_PATH="results metadata path"
-  API_VERSION="API prefix"
-  GUI_COMPONENTS_DETECTION_CROPPING_THRESHOLD="GUI components detection cropping threshold as integer"
-  GAZE_MINIMUM_TIME_STARING="minimum time units user must spend staring at a gui component to take this gui component as a feature from the screenshot"
-  RESULTS_TIMES_FORMAT="results times format (formatted/seconds)"
-  DECISION_TREE_TRAINING_FOLDERNAME="decision tree training phase files foldername"
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