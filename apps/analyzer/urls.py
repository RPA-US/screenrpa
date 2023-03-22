# -*- encoding: utf-8 -*-
"""
Copyright (c) RPA-US
"""

from django.urls import path, re_path
from apps.analyzer import views
from apps.home import views as home

app_name = 'analyzer'

urlpatterns = [
    path('', home.index, name='home'),
    path('list/', views.CaseStudyListView.as_view(), name='casestudy_list'),
    path('new/', views.CaseStudyCreateView.as_view(), name='casestudy_create'),
    path('detail/<int:case_study_id>/', views.CaseStudyDetailView.as_view(), name='casestudy_detail'),
    path('execute/<int:case_study_id>/', views.ExecuteCaseStudyView.as_view(), name='casestudy_execute'),
    path('', views.CaseStudyView.as_view(), name='run-case-study'),
    path('<int:case_study_id>', views.SpecificCaseStudyView.as_view(), name='get-case-study'),
    path('<int:case_study_id>/result', views.ResultCaseStudyView.as_view(), name='get-case-study-result'),
]