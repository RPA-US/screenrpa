"""
WSGI config for ScreenRPA project.

It exposes the WSGI callable as a module-level variable named ``application``.

For more information on this file, see
https://docs.djangoproject.com/en/3.2/howto/deployment/wsgi/
"""

import os
import environ
from django.core.wsgi import get_wsgi_application
env = environ.Env()
environ.Env.read_env()

os.environ.setdefault('DJANGO_SETTINGS_MODULE', env('DJANGO_SETTINGS_MODULE'))

application = get_wsgi_application()
