# -*- encoding: utf-8 -*-
"""
Copyright (c) RPA-US
"""

import os
from decouple import config
from unipath import Path
# from pathlib import Path
import sys
import environ
from django.core.management.utils import get_random_secret_key

# Initialise environment variables
env = environ.Env()
environ.Env.read_env()

DB_NAME =       env('DB_NAME')
DB_HOST =       env('DB_HOST')
DB_PORT =       env('DB_PORT')
DB_USER =       env('DB_USER')
DB_PASSWORD =   env('DB_PASSWORD')


# Build paths inside the project like this: os.path.join(BASE_DIR, ...)
BASE_DIR = Path(__file__).parent
CORE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# SECURITY WARNING: keep the secret key used in production secret!
SECRET_KEY = get_random_secret_key()

# SECURITY WARNING: don't run with debug turned on in production!
DEBUG = config('DEBUG', default=True, cast=bool)
SERVER:str = config('SERVER', default='127.0.0.1')

# load production server from .env
ALLOWED_HOSTS = ['localhost', '127.0.0.1', SERVER]

# Application definition

INSTALLED_APPS = [
    'django.contrib.admin',
    'django.contrib.auth',
    'django.contrib.contenttypes',
    'django.contrib.sessions',
    'django.contrib.messages',
    'django.contrib.staticfiles',
    'rest_framework.authtoken',
    'apps.home',  # Enable the inner home (home)
    'apps.chefboost',
    'apps.analyzer', # Local App
    'apps.featureextraction', # Local App
    'apps.decisiondiscovery', # Local App
    'drf_spectacular',
    'drf_spectacular_sidecar',  # required for Django collectstatic discovery
]

MIDDLEWARE = [
    'django.middleware.security.SecurityMiddleware',
    'whitenoise.middleware.WhiteNoiseMiddleware',
    'django.contrib.sessions.middleware.SessionMiddleware',
    'django.middleware.common.CommonMiddleware',
    'django.middleware.csrf.CsrfViewMiddleware',
    'django.contrib.auth.middleware.AuthenticationMiddleware',
    'django.contrib.messages.middleware.MessageMiddleware',
    'django.middleware.clickjacking.XFrameOptionsMiddleware',
]

ROOT_URLCONF = 'core.urls'
LOGIN_REDIRECT_URL = "home"  # Route defined in home/urls.py
LOGOUT_REDIRECT_URL = "home"  # Route defined in home/urls.py
TEMPLATE_DIR = os.path.join(CORE_DIR, "apps/templates")  # ROOT dir for templates

TEMPLATES = [
    {
        'BACKEND': 'django.template.backends.django.DjangoTemplates',
        'DIRS': [TEMPLATE_DIR],
        'APP_DIRS': True,
        'OPTIONS': {
            'context_processors': [
                'django.template.context_processors.debug',
                'django.template.context_processors.request',
                'django.contrib.auth.context_processors.auth',
                'django.contrib.messages.context_processors.messages',
            ],
        },
    },
]

WSGI_APPLICATION = 'core.wsgi.application'

# Database
# https://docs.djangoproject.com/en/3.0/ref/settings/#databases


DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.postgresql_psycopg2',
        'NAME': DB_NAME,
        'HOST': DB_HOST,
        'PORT': DB_PORT,
        'USER': DB_USER,
        'PASSWORD': DB_PASSWORD,
    }
}

# Password validation
# https://docs.djangoproject.com/en/3.0/ref/settings/#auth-password-validators

AUTH_PASSWORD_VALIDATORS = [
    {
        'NAME': 'django.contrib.auth.password_validation.UserAttributeSimilarityValidator',
    },
    {
        'NAME': 'django.contrib.auth.password_validation.MinimumLengthValidator',
    },
    {
        'NAME': 'django.contrib.auth.password_validation.CommonPasswordValidator',
    },
    {
        'NAME': 'django.contrib.auth.password_validation.NumericPasswordValidator',
    },
]

# Internationalization
# https://docs.djangoproject.com/en/3.0/topics/i18n/

LANGUAGE_CODE = 'en-us'

TIME_ZONE = 'UTC'

USE_I18N = True

USE_L10N = True

USE_TZ = True


# Default primary key field type
# https://docs.djangoproject.com/en/3.2/ref/settings/#default-auto-field

DEFAULT_AUTO_FIELD = 'django.db.models.BigAutoField'

#############################################################

# Static files (CSS, JavaScript, Images)
# https://docs.djangoproject.com/en/1.9/howto/static-files/
STATIC_ROOT = os.path.join(CORE_DIR, 'staticfiles')
STATIC_URL = '/static/'

# Extra places for collectstatic to find static files.
STATICFILES_DIRS = (
    os.path.join(CORE_DIR, 'apps/static'),
)

#############################################################
# ========== 3rd Party Apps: Additional functionality ==========
#############################################################
# - Rest Framework

REST_FRAMEWORK = {
    'DEFAULT_PAGINATION_CLASS': 'rest_framework.pagination.PageNumberPagination',
    'PAGE_SIZE': 10,
    'DEFAULT_SCHEMA_CLASS': 'drf_spectacular.openapi.AutoSchema',
    # 'DATETIME_FORMAT': "%m/%d/%Y %I:%M%P",
    'DEFAULT_AUTHENTICATION_CLASSES': [
        'rest_framework.authentication.TokenAuthentication',
    ],
}

# - Drsf spectacular

SPECTACULAR_SETTINGS = {
    'TITLE': 'RIM API',
    'DESCRIPTION': 'Automatic generation of sintetic UI log in RPA context introducing variability',
    'VERSION': '1.0.0',
    # OTHER SETTINGS
    'SWAGGER_UI_DIST': 'SIDECAR',  # shorthand to use the sidecar instead
    'SWAGGER_UI_FAVICON_HREF': 'SIDECAR',
    'REDOC_DIST': 'SIDECAR',
    # OTHER SETTINGS
}

# Django All Auth config. Add all of this.
EMAIL_BACKEND = 'django.core.mail.backends.console.EmailBackend'

AUTHENTICATION_BACKENDS = (
    "django.contrib.auth.backends.ModelBackend",
    "allauth.account.auth_backends.AuthenticationBackend",
)


# RIM CONFIGURATION
API_VERSION =               env('API_VERSION')
decision_foldername =       env('DECISION_TREE_TRAINING_FOLDERNAME')
cropping_threshold =        int(env('GUI_COMPONENTS_DETECTION_CROPPING_THRESHOLD')) # umbral en el solapamiento de contornos de los gui components al recortarlos
gaze_analysis_threshold =   int(env('GAZE_MINIMUM_TIME_STARING')) # minimum time units user must spend staring at a gui component to take this gui component as a feature from the screenshot
gui_quantity_difference =   int(env('GUI_QUANTITY_DIFFERENCE')) # minimum time units user must spend staring at a gui component to take this gui component as a feature from the screenshot
times_calculation_mode =    env('RESULTS_TIMES_FORMAT') # substitute "formatted" -> get times formatted "%H:%M:%S.%fS" 
metadata_location =         env('METADATA_PATH')
plot_decision_trees =       env('PLOT_DECISION_TREES')
scenario_nested_folder =    env('SCENARIO_NESTED_FOLDER')
several_iterations =        env('SEVERAL_ITERATIONS')
platform_name =                         "RIM"
detection_phase_name =                  "detection"
classification_phase_name =             "classification"
feature_extraction_phase_name =         "feature extraction"
flattening_phase_name =                 "flattening"
decision_model_discovery_phase_name =   "decision model discovery"
FLATTENED_DATASET_NAME =                "flattened_dataset_preprocessed.csv"


# OS SEPARATOR 
operating_system =sys.platform
print("Operating system detected: " + operating_system)
# Element specification filename and path separator (depends on OS)
if "win" in operating_system:
    sep = "\\"
    element_trace = "configuration"+sep+"element_trace.json"
else:
    sep = "/"
    element_trace = "configuration"+sep+"element_trace_linux.json"

# Celery settings
CELERY_BROKER_URL = "redis://localhost:6379"
CELERY_RESULT_BACKEND = "redis://localhost:6379"

# Detect function json conf
FE_EXTRACTORS_FILEPATH = CORE_DIR + sep + "configuration" + sep + "feature_extractors.json"
STATUS_VALUES_ID = CORE_DIR + sep + "configuration" + sep + "status_values_id.json"

# System Default Phases
default_phases = ['ui_elements_detection','noise_filtering','ui_elements_classification','feature_extraction_technique','extract_training_dataset','decision_tree_training']

#############################################################
#############################################################
