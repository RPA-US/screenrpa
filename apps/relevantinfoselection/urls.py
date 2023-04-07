from django.urls import path
from . import views

urlpatterns = [
    path('noise-filtering/list/', views.GazeAnalysisListView.as_view(), name='gaze_analysis_list'),
    path('noise-filtering/new/', views.GazeAnalysisCreateView.as_view(), name='gaze_analysis_create'),
]