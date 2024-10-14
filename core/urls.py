# -*- encoding: utf-8 -*-
"""
Copyright (c) CENIT-ES3
"""

from django.contrib import admin
from django.urls import path, include  # add this
from drf_spectacular.views import SpectacularAPIView, SpectacularSwaggerView, SpectacularRedocView
from .settings import API_VERSION
from apps.analyzer.views import index
from django.conf.urls.i18n import i18n_patterns as _

urlpatterns = [
    path("i18n/", include("django.conf.urls.i18n")),
    path('admin/', admin.site.urls),
    path("", index, name='home'),
    path(API_VERSION+'schema/', SpectacularAPIView.as_view(), name="schema"),
    path(API_VERSION+'docs/', SpectacularSwaggerView.as_view(url_name='schema'), name='swagger-ui'),
    path(API_VERSION+'redoc/', SpectacularRedocView.as_view(url_name='schema'), name='redoc')
]

urlpatterns += _(
    path("", include("apps.authentication.urls")),
    path('case-study/', include("apps.analyzer.urls")),
    path('monitoring/', include("apps.behaviourmonitoring.urls")),
    path('fe/', include("apps.featureextraction.urls")),
    path('pd/', include("apps.processdiscovery.urls")),
    path('dd/', include("apps.decisiondiscovery.urls")),
    path('reporting/', include("apps.reporting.urls")),
    path('notification/', include("apps.notification.urls")),
)
    
