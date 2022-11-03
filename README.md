# RIM tool
Relevance Information Mining tool

## Before run
For development in Windows, make sure you have [Docker](https://docs.docker.com/desktop/install/windows-install/) installed and working. On Linux this is optional, since all Python packages on this project are compatible with it.

If you are going to run this on your local machine (not a container), you need to have [Python](https://www.python.org/downloads/) and [PostgreSQL](https://www.postgresql.org/download/) installed.

If desired, you can create an isolated installation of the project requirements by creating a [virtual environment](https://docs.python.org/3/library/venv.html#:~:text=A%20virtual%20environment%20is%20a,part%20of%20your%20operating%20system.).

## Create a docker container
Clone the repository or download the docker image independently.

Open a terminal on the folder you have downloaded the dockerfile.

Run **`docker build -f Dockerfile.dev --build-arg branch=<banch_name> -t <name> .`** to build the image. By default the branch argument is "main".

Open docker desktop and create a container from the image you just build or run **`docker container create --name testsrim rimtests`** from the command line.

## Configure PostgreSQL
Postgres is not configured by default by the Dockerfile so we will need to do that.

Open your editor or IDE of choice or a terminal and attach the container to it (If you are not using docker just open a terminal).

Run **`service postgresql start`** and **`su postgres`** to start postgres and run a session as the postgres user

Enter psql with **`psql`** and run **`CREATE ROLE "<user>" WITH PASSWORD '<password>' LOGIN CREATEDB;`**. This will be the user you use for the local database

Now create the database you will use with the django application. For that, exit the psql session and run **`psql postgres <user>`** and then **`CREATE DATABASE <database_name>;`** to create the database

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