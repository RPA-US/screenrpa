# -*- encoding: utf-8 -*-
"""
Copyright (c) RPA-US
"""

from django.contrib import admin
from django.urls import path, include  # add this
from drf_spectacular.views import SpectacularAPIView, SpectacularSwaggerView, SpectacularRedocView
from .settings import API_VERSION
from apps.analyzer.views import index

urlpatterns = [
    path('admin/', admin.site.urls),          # Django admin route
    path("", index, name='home'),
    path("", include("apps.authentication.urls")), # Auth routes - login / register
    path('case-study/', include("apps.analyzer.urls")),
    path('fe/', include("apps.featureextraction.urls")),
    path('dd/', include("apps.decisiondiscovery.urls")),
    path(API_VERSION+'schema/', SpectacularAPIView.as_view(), name="schema"),
    path(API_VERSION+'docs/', SpectacularSwaggerView.as_view(url_name='schema'), name='swagger-ui'),
    path(API_VERSION+'redoc/', SpectacularRedocView.as_view(url_name='schema'), name='redoc')
]
