from django.urls import path
from . import views

app_name = 'behaviourmonitoring'

urlpatterns = [
    path('list/<int:case_study_id>/', views.MonitoringListView.as_view(), name='monitoring_list'),
    path('new/<int:case_study_id>/', views.MonitoringCreateView.as_view(), name='monitoring_create'),
    path('detail/<int:monitoring_id>/', views.MonitoringDetailView.as_view(), name='monitoring_detail'),
    path('active/', views.set_as_active, name='monitoring_set_as_active'),
    path('delete/', views.delete_monitoring, name='monitoring_delete')
]