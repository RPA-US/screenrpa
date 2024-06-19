
from django.urls import path
from . import views

app_name = 'notification'

urlpatterns = [
    path('', views.get_notifications, name='notification_list'),
    path('mark_as_read/', views.mark_as_read, name='mark_as_read'),
    path('mark_all_as_read/', views.mark_all_as_read, name='mark_as_read'),
    path('delete/', views.delete_notification, name='delete'),
]