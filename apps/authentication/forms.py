# -*- encoding: utf-8 -*-
"""
Copyright (c) RPA-US
"""

from django import forms
from django.contrib.auth.forms import UserCreationForm
from django.contrib.auth.models import User


class LoginForm(forms.Form):
    username = forms.CharField(
        widget=forms.TextInput(
            attrs={
                "placeholder": "Username",
                "class": "form-control"
            }
        ))
    password = forms.CharField(
        widget=forms.PasswordInput(
            attrs={
                "placeholder": "Password",
                "class": "form-control"
            }
        ))


class SignUpForm(UserCreationForm):
    firstname = forms.CharField(
        widget=forms.TextInput(
            attrs={
                "placeholder": "First Name",
                "class": "form-control"
            }
        ))
    lastname = forms.CharField(
        widget=forms.TextInput(
            attrs={
                "placeholder": "Last Name",
                "class": "form-control"
            }
        ))
    username = forms.CharField(
        widget=forms.TextInput(
            attrs={
                "placeholder": "Username",
                "class": "form-control"
            }
        ))
    email = forms.EmailField(
        widget=forms.EmailInput(
            attrs={
                "placeholder": "Email",
                "class": "form-control"
            }
        ))
    password1 = forms.CharField(
        widget=forms.PasswordInput(
            attrs={
                "placeholder": "Password",
                "class": "form-control"
            }
        ))
    password2 = forms.CharField(
        widget=forms.PasswordInput(
            attrs={
                "placeholder": "Password check",
                "class": "form-control"
            }
        ))

    class Meta:
        model = User
        fields = ('username', 'email', 'password1', 'password2')

class UserForm(forms.ModelForm):
    class Meta:
        model = User
        exclude = ('is_staff', 'is_active', 'is_superuser')
        fields = ('username', 'email', 'first_name', 'last_name')
        widgets = {
            "first_name": forms.TextInput(
                attrs={
                    "id": "input-first_name",
                    "class": "form-control",
                    "placeholder": "First name"
                    }
            ),
            "last_name": forms.TextInput(
                attrs={
                    "id": "input-last_name",
                    "class": "form-control",
                    "placeholder": "Last Name"
                    }
            ),
            "username": forms.TextInput(
                attrs={
                    "id": "input-username",
                    "class": "form-control",
                    "placeholder": "Username"
                    }
            ),
            "email": forms.EmailInput(
                attrs={
                    "id": "input-email",
                    "class": "form-control",
                    "placeholder": "Email"
                    }
            )
            }