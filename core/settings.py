# -*- encoding: utf-8 -*-
"""
Copyright (c) RPA-US
"""

import os
from decouple import config
from unipath import Path
import sys
import environ
from django.core.management.utils import get_random_secret_key
import logging.config
from django.utils.translation import gettext_lazy as _

#===================================================================================================
#===================================================================================================

# Initialise environment variables
env = environ.Env()
environ.Env.read_env()

# Framework Environment Variables
DB_NAME =                       env('DB_NAME')
DB_HOST =                       env('DB_HOST')
DB_PORT =                       env('DB_PORT')
DB_USER =                       env('DB_USER')
DB_PASSWORD =                   env('DB_PASSWORD')
API_VERSION =                   env('API_VERSION')
ACTIVE_CELERY =                 config('DISABLE_MULTITHREADING', default=False, cast=bool)
SCENARIO_NESTED_FOLDER =        config('SCENARIO_NESTED_FOLDER', default=False, cast=bool)
METADATA_LOCATION =             env('METADATA_PATH')
FIXATION_DURATION_THRESHOLD =   int(env('FIXATION_DURATION_THRESHOLD')) # minimum time units user must spend staring at a gui component to take this gui component as a feature from the screenshot
CROPPING_THRESHOLD =            int(env('GUI_COMPONENTS_DETECTION_CROPPING_THRESHOLD')) # umbral en el solapamiento de contornos de los gui components al recortarlos
GUI_QUANTITY_DIFFERENCE =       int(env('GUI_QUANTITY_DIFFERENCE')) # minimum time units user must spend staring at a gui component to take this gui component as a feature from the screenshot
FLATTENED_DATASET_NAME =        env('FLATTENED_DATASET_NAME')
SEVERAL_ITERATIONS =            int(env('DECISION_TREE_TRAINING_ITERATIONS'))
DECISION_FOLDERNAME =           env('DECISION_TREE_TRAINING_FOLDERNAME')
PLOT_DECISION_TREES =           config('PLOT_DECISION_TREES', default=False, cast=bool)

# Framework Phases names
PLATFORM_NAME =                             "  SCREEN RPA"
MONITORING_PHASE_NAME =                     _("monitoring")
INFO_PREFILTERING_PHASE_NAME =              _("preselection")
DETECTION_PHASE_NAME =                      _("detection")
CLASSIFICATION_PHASE_NAME =                 _("classification")
INFO_POSTFILTERING_PHASE_NAME =             _("selection")
SINGLE_FEATURE_EXTRACTION_PHASE_NAME =      _("feature extraction")
FLATTENING_PHASE_NAME =                     _("flattening")
AGGREGATE_FEATURE_EXTRACTION_PHASE_NAME =   _("aggreate feature extraction")
DECISION_MODEL_DISCOVERY_PHASE_NAME =       _("decision model discovery")
ENRICHED_LOG_SUFFIX =                       "_enriched"

# System Default Phases
DEFAULT_PHASES = ['monitoring', 'prefilters', 'ui_elements_detection', 'ui_elements_classification', 'postfilters', 'feature_extraction_technique', 
                  'process_discovery', 'extract_training_dataset', 'feature_extraction_technique', 'decision_tree_training']
# DEFAULT_PHASES = ['monitoring','info_prefiltering','ui_elements_detection','ui_elements_classification','info_postfiltering','process_discovery',
#                  'feature_extraction_technique','extract_training_dataset','aggregate_features_as_dataset_columns','decision_tree_training']
PHASES_OBJECTS = ['Monitoring','Prefilters','UIElementsDetection','UIElementsClassification','Postfilters','FeatureExtractionTechnique','ProcessDiscovery','ExtractTrainingDataset','DecisionTreeTraining']
MONITORING_IMOTIONS_NEEDED_COLUMNS = ["CoorX","CoorY","EventType","NameApp","Screenshot"]

#===================================================================================================
#===================================================================================================

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

#===================================================================================================
#===================================================================================================

# Application definition
INSTALLED_APPS = [
    'django.contrib.admin',
    'django.contrib.auth',
    'django.contrib.contenttypes',
    'django.contrib.sessions',
    'django.contrib.messages',
    'django.contrib.staticfiles',
    'django_extensions',
    'rest_framework.authtoken',
    'private_storage',
    'apps.chefboost',
    'apps.authentication',
    'apps.analyzer', # Local App
    'apps.behaviourmonitoring', # Local App
    'apps.featureextraction', # Local App
    'apps.processdiscovery', # Local App
    'apps.decisiondiscovery', # Local App
    'apps.reporting', # Local App
    'drf_spectacular', # Swagger
    'drf_spectacular_sidecar',  # Swagger. required for Django collectstatic discovery
]

MIDDLEWARE = [
    'django.middleware.security.SecurityMiddleware',
    'whitenoise.middleware.WhiteNoiseMiddleware',
    'django.contrib.sessions.middleware.SessionMiddleware',
    'django.middleware.locale.LocaleMiddleware',
    'django.middleware.common.CommonMiddleware',
    'django.middleware.csrf.CsrfViewMiddleware',
    'django.contrib.auth.middleware.AuthenticationMiddleware',
    'django.contrib.messages.middleware.MessageMiddleware',
    'django.middleware.clickjacking.XFrameOptionsMiddleware',
]


#===================================================================================================
#===================================================================================================

ROOT_URLCONF = 'core.urls'
LOGIN_REDIRECT_URL = "home"  # Route defined in core/urls.py
LOGOUT_REDIRECT_URL = "home"  # Route defined in core/urls.py
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
                'django.template.context_processors.i18n',
                'django.contrib.auth.context_processors.auth',
                'django.contrib.messages.context_processors.messages',
            ],
            'libraries': {
                'extras': 'apps.templatetags.extras',
            }
        },
    },
]

WSGI_APPLICATION = 'core.wsgi.application'


#===================================================================================================
#===================================================================================================

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

#===================================================================================================
#===================================================================================================

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

USE_I18N = True
USE_L10N = True
LANGUAGE_CODE = 'en'
LANGUAGES = [
    ('en', _('English')),
    ('es', _('Spanish'))
]
LOCALE_PATHS = [
    os.path.join(BASE_DIR, 'locale'),
    os.path.join(BASE_DIR, 'apps', 'analyzer', 'locale'),
    os.path.join(BASE_DIR, 'apps', 'authentication', 'locale'),
    os.path.join(BASE_DIR, 'apps', 'behaviourmonitoring', 'locale'),
    os.path.join(BASE_DIR, 'apps', 'chefboost', 'locale'),
    os.path.join(BASE_DIR, 'apps', 'decisiondiscovery', 'locale'),
    os.path.join(BASE_DIR, 'apps', 'featureextraction', 'locale'),
    os.path.join(CORE_DIR, 'locale'),
]

USE_TZ = True
TIME_ZONE = 'UTC'

# Default primary key field type
# https://docs.djangoproject.com/en/3.2/ref/settings/#default-auto-field

DEFAULT_AUTO_FIELD = 'django.db.models.BigAutoField'

#===================================================================================================
#===================================================================================================

# Static files (CSS, JavaScript, Images)
# https://docs.djangoproject.com/en/1.9/howto/static-files/
STATIC_ROOT = os.path.join(CORE_DIR, 'staticfiles')
STATIC_URL = '/static/'

# Extra places for collectstatic to find static files.
STATICFILES_DIRS = (
    os.path.join(CORE_DIR, 'apps/static'),
)


#===================================================================================================
#===================================================================================================
# ========== 3rd Party Apps: Additional functionality ==============================================

# Django Rest Framework
REST_FRAMEWORK = {
    'DEFAULT_PAGINATION_CLASS': 'rest_framework.pagination.PageNumberPagination',
    'PAGE_SIZE': 10,
    'DEFAULT_SCHEMA_CLASS': 'drf_spectacular.openapi.AutoSchema',
    # 'DATETIME_FORMAT': "%m/%d/%Y %I:%M%P",
    'DEFAULT_AUTHENTICATION_CLASSES': [
        'rest_framework.authentication.TokenAuthentication',
    ],
}

# Swagger Drsf spectacular
SPECTACULAR_SETTINGS = {
    'TITLE': 'SCREEN RPA API',
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
EMAIL_HOST = 'smtp.gmail.com'
EMAIL_PORT = 587
EMAIL_HOST_USER = env('EMAIL_HOST_USER')
EMAIL_HOST_PASSWORD = env('EMAIL_HOST_PASSWORD')
EMAIL_USE_TLS = True

AUTHENTICATION_BACKENDS = (
    "django.contrib.auth.backends.ModelBackend",
    "allauth.account.auth_backends.AuthenticationBackend",
)

# Private storage config
PRIVATE_STORAGE_ROOT = 'media'
PRIVATE_STORAGE_AUTH_FUNCTION = 'apps.analyzer.permissions.allow_staff'


# Celery settings
CELERY_BROKER_URL = "redis://localhost:6379"
CELERY_RESULT_BACKEND = "redis://localhost:6379"

#===================================================================================================
#===================================================================================================

# Configure logging
LOGGING = {
    'version': 1,
    'disable_existing_loggers': False,
    'handlers': {
        'file': {
            'level': 'INFO',
            'class': 'logging.FileHandler',
            'filename': 'screenrpa.log',
        },
    },
    'loggers': {
        '': {
            'handlers': ['file'],
            'level': 'INFO',
        },
    },
}

# Apply the logging configuration
logging.config.dictConfig(LOGGING)

#===================================================================================================
#===================================================================================================

# Operating System Separator
operating_system =sys.platform
print("Operating system detected: " + operating_system)
# Element specification filename and path separator (depends on OS)
if "win" in operating_system:
    sep = "\\"
    element_trace = CORE_DIR + sep + "configuration"+sep+"element_trace.json"
else:
    sep = "/"
    element_trace = CORE_DIR + sep + "configuration"+sep+"element_trace_linux.json"

#===================================================================================================
#===================================================================================================

# Configuration JSON files Paths
SINGLE_FE_EXTRACTORS_FILEPATH =     CORE_DIR + sep + "configuration" + sep + "single_feature_extractors.json"
AGGREGATE_FE_EXTRACTORS_FILEPATH =  CORE_DIR + sep + "configuration" + sep + "aggreate_feature_extractors.json"
MODELS_CLASSES_FILEPATH =           CORE_DIR + sep + "configuration" + sep + "models_classes.json"
STATUS_VALUES_ID =                  CORE_DIR + sep + "configuration" + sep + "status_values_id.json"
CDLR =                              CORE_DIR + sep + "configuration" + sep + "cdlr.json"