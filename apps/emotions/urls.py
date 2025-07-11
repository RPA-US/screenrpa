from django.urls import path
from . import views


urlpatterns = [
    path('emotions/',       views.emotion_home,     name='emotion_home'),
    path('emotions/start/', views.start_record,    name='emotion_start'),
    path('emotions/stop/',  views.stop_record,     name='emotion_stop'),
    path('emotions/download/', views.download_record, name='emotion_download'),
]