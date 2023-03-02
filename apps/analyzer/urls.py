# -*- encoding: utf-8 -*-
"""
Copyright (c) RPA-US
"""

from django.urls import path, re_path
from apps.analyzer import views

app_name = 'analyzer'

urlpatterns = [
    path('list/', views.CaseStudyListView.as_view(), name='casestudy_list'),
    path('new/', views.CaseStudyCreateView.as_view(), name='casestudy_create'),
    path('detail/<int:case_study_id>/', views.CaseStudyCreateView.as_view(), name='casestudy_create'),
    path('', views.CaseStudyView.as_view(), name='run-case-study'),
    path('<int:case_study_id>', views.SpecificCaseStudyView.as_view(), name='get-case-study'),
    path('<int:case_study_id>/result', views.ResultCaseStudyView.as_view(), name='get-case-study-result'),
]