from django.urls import path
from . import views

app_name = 'behaviourmonitoring'

urlpatterns = [
    path('list/<int:case_study_id>/', views.MonitoringListView.as_view(), name='monitoring_list'),
    path('new/', views.MonitoringCreateView.as_view(), name='monitoring_create'),
]