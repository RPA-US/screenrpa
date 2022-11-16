#!/bin/bash
/rim/venv/bin/python /rim/manage.py makemigrations analyzer decisiondiscovery featureextraction
/rim/venv/bin/python /rim/manage.py migrate
/rim/venv/bin/python /rim/manage.py collectstatic
/rim/venv/bin/python /rim/manage.py runserver 0.0.0.0:8000