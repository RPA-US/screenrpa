# -*- encoding: utf-8 -*-
"""
Copyright (c) CENIT-ES3
"""

from django.urls import include, path
from .views import login_view, register_user, edit_user
from django.contrib.auth.views import LogoutView
from rest_framework.authtoken import views


urlpatterns = [
    path('api-token-auth/', views.obtain_auth_token),
    path('login/', login_view, name="login"),
    path('register/', register_user, name="register"),
    path('profile/', edit_user, name="edit-user"),
    path("logout/", LogoutView.as_view(), name="logout")
]
