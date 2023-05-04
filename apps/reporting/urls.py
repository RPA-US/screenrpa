from django.urls import path
from . import views

app_name = 'reporting'

urlpatterns = [
    path('pdd/list/', views.ReportListView.as_view(), name='report_list'),
    path('pdd/download/<int:report_id>', views.ReportListView.as_view(), name='report_download'),
    path('pdd/generate/<int:case_study_id>', views.ReportGenerateView.as_view(), name='report_generate'),
]