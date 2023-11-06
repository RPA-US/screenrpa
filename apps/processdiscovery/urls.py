from django.urls import path
from . import views

app_name = 'processdiscovery'

urlpatterns = [
    path('bpmn/detail/<int:case_study_id>/<int:process_discovery_id>/', views.ProcessDiscoveryDetailView.as_view(), name='processdiscovery_detail'),
    path('bpmn/active/', views.set_as_process_discovery_active, name='processdiscovery_set_as_active'),
    path('bpmn/delete/', views.delete_process_discovery, name='processdiscovery_delete'),
    path('bpmn/list/<int:case_study_id>/', views.ProcessDiscoveryListView.as_view(), name='processdiscovery_list'),
    path('bpmn/new/<int:case_study_id>/', views.ProcessDiscoveryCreateView.as_view(), name='processdiscovery_create'),
]