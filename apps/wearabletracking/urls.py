from django.urls import path
from . import views

app_name = 'wearabletracking'

urlpatterns = [
    path('wearable/', views.wearable_home, name='wearable_home'),
    path('authorize/', views.authorize, name='authorize'),
    path('callback/', views.callback, name='callback'),
    path('rango-fechas/', views.exportar_datos_fitbit, name='exportar_datos_fitbit'),
    path('analytics/', views.analytics, name='fitbit_analytics'),

]