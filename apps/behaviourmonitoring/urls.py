from django.urls import path
from . import views

urlpatterns = [
    path('gaze-analysis/list/', views.GazeAnalysisListView.as_view(), name='gaze_analysis_list'),
    path('gaze-analysis/new/', views.GazeAnalysisCreateView.as_view(), name='gaze_analysis_create'),
]