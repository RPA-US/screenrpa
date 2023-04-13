# -*- encoding: utf-8 -*-
"""
Copyright (c) RPA-US
"""

from django.urls import path, re_path
from apps.analyzer import views

app_name = 'analyzer'

urlpatterns = [
    # The home page
    # Matches any html file
    re_path(r'^.*\.html*', views.pages, name='pages'),
    path('list/', views.CaseStudyListView.as_view(), name='casestudy_list'),
    path('new/', views.CaseStudyCreateView.as_view(), name='casestudy_create'),
    path('detail/<int:case_study_id>/', views.CaseStudyDetailView.as_view(), name='casestudy_detail'),
    path('execute/', views.executeCaseStudy, name='casestudy_execute'),
    path('delete/', views.deleteCaseStudy, name='casestudy_delete'),
    path('api/', views.CaseStudyView.as_view(), name='run-case-study'),
    path('<int:case_study_id>', views.SpecificCaseStudyView.as_view(), name='get-case-study'),
    path('<int:case_study_id>/result', views.ResultCaseStudyView.as_view(), name='get-case-study-result'),
]