"""
RIM URL Configuration
"""
from .production import API_VERSION
from django.contrib import admin
from django.urls import include, path
import analyzer
from drf_spectacular.views import SpectacularAPIView, SpectacularSwaggerView, SpectacularRedocView

urlpatterns = [
    path(API_VERSION+'admin/', admin.site.urls),
    path(API_VERSION+'analyzer/', include('analyzer.urls')),
    path(API_VERSION+'schema/', SpectacularAPIView.as_view(), name="schema"),
    path(API_VERSION+'docs/', SpectacularSwaggerView.as_view(url_name='schema'), name='swagger-ui'),
    path(API_VERSION+'redoc/', SpectacularRedocView.as_view(url_name='schema'), name='redoc')
]
