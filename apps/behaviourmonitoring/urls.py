from django.urls import path
from . import views

urlpatterns = [
    path('gaze-analysis/list/', views.MonitoringListView.as_view(), name='monitoring_list'),
    path('gaze-analysis/new/', views.MonitoringCreateView.as_view(), name='monitoring_create'),
]