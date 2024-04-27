from django.urls import path
from . import views

app_name = 'reporting'

urlpatterns = [
    path('pdd/list/<int:case_study_id>/', views.ReportListView.as_view(), name='report_list'),
    path('pdd/download/<int:report_id>', views.ReportListView.as_view(), name='report_download'),
    path('pdd/generate/<int:execution_id>', views.ReportCreateView.as_view(), name='report_generate'),
]