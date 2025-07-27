from django.urls import path
from . import views

app_name = 'wearabletracking'

urlpatterns = [
    path('wearable/', views.wearable_home, name='wearable_home'),
    path('authorize/', views.authorize, name='authorize'),
    path('callback/', views.callback, name='callback'),
    path('rango-fechas/', views.exportar_datos_fitbit, name='exportar_datos_fitbit'),
    path('analytics/', views.analytics, name='fitbit_analytics'),
    path('fitbit/callback/', views.callback, name='fitbit_callback'),
    path('logout/', views.fitbit_logout, name='fitbit_logout'),
    
    # NUEVA FASE: BiometricAnalysis
    path('biometric-analysis/list/<int:case_study_id>/', views.biometric_config_list, name='biometric_config_list'),
    path('biometric-analysis/new/<int:case_study_id>/', views.biometric_config_create, name='biometric_config_create'),
    path('biometric-analysis/detail/<int:config_id>/', views.biometric_config_detail, name='biometric_config_detail'),
    path('biometric-analysis/activate/<int:config_id>/', views.biometric_config_activate, name='biometric_config_activate'),
    path('biometric-analysis/delete/<int:config_id>/', views.biometric_config_delete, name='biometric_config_delete'),
    path('biometric-analysis/edit/<int:config_id>/', views.biometric_config_edit, name='biometric_config_edit'),

    # Reportes biométricos
    path('biometric-report/list/<int:execution_id>/', views.biometric_report_list, name='biometric_report_list'),
    path('biometric-report/detail/<int:report_id>/', views.biometric_report_detail, name='biometric_report_detail'),
    path('biometric-report/download/<int:report_id>/', views.biometric_report_download, name='biometric_report_download'),
]