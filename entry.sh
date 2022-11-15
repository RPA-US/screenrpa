#!/bin/bash
/rim/venv/bin/python /rim/manage.py makemigrations analyzer decisiondiscovery featureextraction
/rim/venv/bin/python /rim/manage.py migrate
/rim/venv/bin/python -m gunicorn rim.wsgi:application -b 0.0.0.0:8000