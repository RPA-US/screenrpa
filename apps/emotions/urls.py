from django.urls import path
from . import views


urlpatterns = [
    path('',       views.emotion_home,     name='emotion_home'),
    path('start/', views.start_record,    name='emotion_start'),
    path('stop/',  views.stop_record,     name='emotion_stop'),
    path('download/', views.download_record, name='emotion_download'),
    path('current/', views.get_current_emotion, name='get_current_emotion'),
    path('video_feed/', views.video_feed, name='video_feed'),
    path('camera_test/', views.camera_test, name='camera_test'),  # Nueva ruta

]