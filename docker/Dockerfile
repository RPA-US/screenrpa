FROM python:3.10.7-slim-buster
# ARG postgresuser=root
# ARG postgrespassword=root

# set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update
RUN apt install -y python3.10-venv python3.10-dev python3.10-distutils python3.10-lib2to3 python3.10-gdbm

# Install rim system depenencies
RUN apt-get install -y gcc git postgresql-server-dev-all musl-dev libffi-dev cmake python-tk g++ ffmpeg libsm6 libxext6 redis
RUN apt-get install -y postgresql postgresql-client
# Allows docker to cache installed dependencies between builds

# Copy the project files
WORKDIR /screenrpa
COPY . .

# Installs python dependencies
RUN python3.10 -m venv venv
RUN ./venv/bin/python -m pip install --upgrade pip
RUN ./venv/bin/python -m pip install --no-cache-dir -r requirements.txt
RUN ./venv/bin/python -m pip install tensorflow==2.10.0
RUN ./venv/bin/python -m pip install transformers

# Install graphviz
RUN apt-get update && apt-get install -y graphviz graphviz-dev
# Install libreoffice
RUN apt-get install libreoffice-core-nogui libreoffice-writer-nogui --no-install-recommends --no-install-suggests

# Make django migrations
COPY ./docker/.env ./core/.env
RUN ./venv/bin/python manage.py makemigrations apps_analyzer apps_behaviourmonitoring apps_decisiondiscovery apps_featureextraction apps_processdiscovery apps_reporting

# Internationalization
RUN apt-get install -y gettext
RUN ./venv/bin/python manage.py compilemessages
RUN ./venv/bin/python manage.py collectstatic --noinput