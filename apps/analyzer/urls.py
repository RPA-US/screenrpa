# -*- encoding: utf-8 -*-
"""
Copyright (c) RPA-US
"""

from django.urls import path, include, re_path
from apps.analyzer import views
import private_storage.urls

app_name = 'analyzer'

urlpatterns = [
    # The home page
    # Matches any html file
    re_path(r'^.*\.html*', views.pages, name='pages'),
    path('list/', views.CaseStudyListView.as_view(), name='casestudy_list'),
    path('files/list/', views.exp_files, name='files_list'),
    path('files/download/<int:case_study_id>/', views.exp_file_download, name='file_download'),
    path('new/', views.CaseStudyCreateView.as_view(), name='casestudy_create'),
    path('detail/<int:case_study_id>/', views.CaseStudyDetailView.as_view(), name='casestudy_detail'),
    path('execute/', views.executeCaseStudy, name='casestudy_execute'),
    path('delete/', views.deleteCaseStudy, name='casestudy_delete'),
    path('api/', views.CaseStudyView.as_view(), name='run-case-study'),
    path('<int:case_study_id>', views.SpecificCaseStudyView.as_view(), name='get-case-study'),
    path('<int:case_study_id>/result', views.ResultCaseStudyView.as_view(), name='get-case-study-result'),
    re_path('^private-data/', include(private_storage.urls)),
    # Executions
    path('execution/list/', views.ExecutionListView.as_view(), name='execution_list'),
    path('execution/delete/', views.deleteExecution, name='execution_delete'),
    path('execution/detail/<int:execution_id>/', views.ExecutionDetailView.as_view(), name='execution_detail'),
    ##
    # Phases Executions results
    path('execution/monitoring_result/<int:execution_id>/', views.MonitoringResultDetailView.as_view(), name='monitoring_result'),
    path('execution/featureExtraction_result/<int:execution_id>/', views.FeatureExtractionResultDetailView.as_view(), name='featureExtraction_result'),
    path('execution/extractTrainingDataset_result/<int:execution_id>/', views.ExtractTrainingDatasetResultDetailView.as_view(), name='extractTrainingDataset_result'),
    path('execution/process_discovery_result/<int:execution_id>/', views.mostrar_diagrama.as_view(), name='process_discovery_result'),
    


]

