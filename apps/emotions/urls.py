from django.urls import path
from . import views

app_name = 'emotions'

urlpatterns = [
    path('emotions/', views.emotion_home, name='emotion_home'),

]