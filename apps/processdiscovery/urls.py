from django.urls import path
from . import views

app_name = 'processdiscovery'

urlpatterns = [
    path('bpmn/list/<int:case_study_id>/', views.ProcessDiscoveryListView.as_view(), name='processdiscovery_list'),
    path('bpmn/new/<int:case_study_id>/', views.ProcessDiscoveryCreateView.as_view(), name='processdiscovery_create'),
]