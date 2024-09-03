FROM nvidia/cuda:11.8.0-devel-ubuntu20.04

# set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1
ENV DEBIAN_FRONTEND=noninteractive

# Install python 3.10
RUN apt-get update
RUN apt-get install software-properties-common -y
RUN add-apt-repository ppa:deadsnakes/ppa -y
RUN apt-get update
RUN apt-get install python3.10 -y
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

# Upgrades libstdc++6
RUN add-apt-repository ppa:ubuntu-toolchain-r/test -y
RUN apt update
RUN apt upgrade libstdc++6 -y 

# Installs cudnn8
RUN apt install libcudnn8 libcudnn8-dev -y

# Install graphviz
RUN apt-get update && apt-get install -y graphviz graphviz-dev
# Install libreoffice
RUN apt-get install libreoffice-core-nogui libreoffice-writer-nogui --no-install-recommends --no-install-suggests

# Install latex dependencies for pandoc
# Make django migrations
COPY ./docker/.env ./core/.env
RUN ./venv/bin/python manage.py makemigrations apps_analyzer apps_behaviourmonitoring apps_decisiondiscovery apps_featureextraction apps_processdiscovery apps_reporting

# Internationalization
RUN apt-get install -y gettext
RUN ./venv/bin/python manage.py compilemessages
RUN ./venv/bin/python manage.py collectstatic --noinput
