# Temporary until framework is updated to a newer version of python
FROM python:3.10.19-trixie

# set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1
ENV DEBIAN_FRONTEND=noninteractive

# Install rim system depenencies
RUN apt-get update
RUN apt-get install -y gcc git postgresql-server-dev-all musl-dev libffi-dev cmake g++ ffmpeg libsm6 libxext6 redis redis-server
RUN apt-get install -y postgresql postgresql-client

# Install graphviz
RUN apt-get update && apt-get install -y graphviz graphviz-dev
# Install libreoffice
RUN apt-get install -y libreoffice-core-nogui libreoffice-writer-nogui --no-install-recommends --no-install-suggests

# Copy the project files
WORKDIR /screenrpa
COPY . .

# Installs python dependencies
RUN python -m venv venv
RUN ./venv/bin/python -m pip install --upgrade pip
RUN ./venv/bin/python -m pip install --no-cache-dir -r requirements.txt
RUN ./venv/bin/python -m pip install tensorflow==2.10.0
RUN ./venv/bin/python -m pip install transformers

# Internationalization/tools needed at runtime
RUN apt-get install -y gettext

COPY ./docker/docker-entrypoint.sh /usr/local/bin/docker-entrypoint.sh
RUN chmod +x /usr/local/bin/docker-entrypoint.sh

ENTRYPOINT ["/usr/local/bin/docker-entrypoint.sh"]