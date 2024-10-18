from django.urls import path
from . import views

app_name = 'behaviourmonitoring'

urlpatterns = [
    path('list/<int:case_study_id>/', views.MonitoringListView.as_view(), name='monitoring_list'),
    path('new/<int:case_study_id>/', views.MonitoringCreateView.as_view(), name='monitoring_create'),
    path('detail/<int:case_study_id>/<int:monitoring_id>/', views.MonitoringDetailView.as_view(), name='monitoring_detail'),
    path('ex/detail/<int:execution_id>/<int:monitoring_id>/', views.MonitoringDetailView.as_view(), name='monitoring_detail-execution'),

    path('active/', views.set_as_active, name='monitoring_set_as_active'),
    path('inactive/', views.set_as_inactive, name='monitoring_set_as_inactive'),
    path('delete/', views.delete_monitoring, name='monitoring_delete'),
    path('result/<int:execution_id>/', views.MonitoringResultDetailView.as_view(), name='monitoring_result'),
]