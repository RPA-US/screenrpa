# -*- encoding: utf-8 -*-
"""
Copyright (c) CENIT-ES3
"""

from django import forms
from django.contrib.auth.forms import UserCreationForm
from django.contrib.auth.models import User
from django.utils.translation import gettext_lazy as _


class LoginForm(forms.Form):
    username = forms.CharField(
        widget=forms.TextInput(
            attrs={
                "placeholder": _("Username"),
                "class": "form-control"
            }
        ))
    password = forms.CharField(
        widget=forms.PasswordInput(
            attrs={
                "placeholder": _("Password"),
                "class": "form-control"
            }
        ))


class SignUpForm(UserCreationForm):
    firstname = forms.CharField(
        widget=forms.TextInput(
            attrs={
                "placeholder": _("First Name"),
                "class": "form-control"
            }
        ))
    lastname = forms.CharField(
        widget=forms.TextInput(
            attrs={
                "placeholder": _("Last Name"),
                "class": "form-control"
            }
        ))
    username = forms.CharField(
        widget=forms.TextInput(
            attrs={
                "placeholder": _("Username"),
                "class": "form-control"
            }
        ))
    email = forms.EmailField(
        widget=forms.EmailInput(
            attrs={
                "placeholder": _("Email"),
                "class": "form-control"
            }
        ))
    password1 = forms.CharField(
        widget=forms.PasswordInput(
            attrs={
                "placeholder": _("Password"),
                "class": "form-control"
            }
        ))
    password2 = forms.CharField(
        widget=forms.PasswordInput(
            attrs={
                "placeholder": _("Password check"),
                "class": "form-control"
            }
        ))

    class Meta:
        model = User
        fields = ('username', 'email', 'password1', 'password2')
        labels = {
            "username": _("Username"),
            "email": _("Email"),
            "password1": _("Password"),
            "password2": _("Password check")
        }

class UserForm(forms.ModelForm):
    class Meta:
        model = User
        exclude = ('is_staff', 'is_active', 'is_superuser')
        fields = ('username', 'email', 'first_name', 'last_name')
        labels = {
            "username": _("Username"),
            "email": _("Email"),
            "first_name": _("First name"),
            "last_name": _("Last name")
        }
        widgets = {
            "first_name": forms.TextInput(
                attrs={
                    "id": "input-first_name",
                    "class": "form-control",
                    "placeholder": _("First name")
                    }
            ),
            "last_name": forms.TextInput(
                attrs={
                    "id": "input-last_name",
                    "class": "form-control",
                    "placeholder": _("Last Name")
                    }
            ),
            "username": forms.TextInput(
                attrs={
                    "id": "input-username",
                    "class": "form-control",
                    "placeholder": _("Username")
                    }
            ),
            "email": forms.EmailInput(
                attrs={
                    "id": "input-email",
                    "class": "form-control",
                    "placeholder": _("Email")
                    }
            )
            }