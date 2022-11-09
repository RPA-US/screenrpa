FROM python:3.10.7-slim-buster

RUN apt-get update
RUN apt-get install -y gcc postgresql-server-dev-all musl-dev libffi-dev cmake python-tk g++ ffmpeg libsm6 libxext6

WORKDIR /rim

# Installs python dependencies
COPY requirements.txt .
RUN /usr/local/bin/python -m venv venv
RUN ./venv/bin/python -m pip install --upgrade pip
RUN ./venv/bin/python -m pip install --no-cache-dir gunicorn
RUN ./venv/bin/python -m pip install --no-cache-dir -r requirements.txt

# Copies the rest of the code
COPY . .

RUN chmod +x ./entry.sh