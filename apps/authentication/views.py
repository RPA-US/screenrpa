# -*- encoding: utf-8 -*-
"""
Copyright (c) CENIT-ES3
"""

# Create your views here.
from django.shortcuts import render, redirect
from django.contrib.auth import authenticate, login
from .forms import LoginForm, SignUpForm, UserForm
from django.contrib.auth.decorators import login_required
from django.urls import reverse
from apps.analyzer.models import CaseStudy
from django.http import HttpResponseRedirect
from django.utils.translation import gettext_lazy as _

def login_view(request):
    form = LoginForm(request.POST or None)

    msg = None

    if request.method == "POST":

        if form.is_valid():
            username = form.cleaned_data.get("username")
            password = form.cleaned_data.get("password")
            user = authenticate(username=username, password=password)
            if user is not None:
                login(request, user)
                if request.GET.get("next") is not None:
                    return redirect(request.GET.get("next"))
                return redirect("/")
            else:
                msg = _('Invalid credentials')
        else:
            msg = _('Error validating the form')

    return render(request, "accounts/login.html", {"form": form, "msg": msg})


def register_user(request):
    msg = None
    success = False

    if request.method == "POST":
        form = SignUpForm(request.POST)
        if form.is_valid():
            form.save()
            firstname = form.cleaned_data.get("firstname")
            lastname = form.cleaned_data.get("lastname")
            username = form.cleaned_data.get("username")
            raw_password = form.cleaned_data.get("password1")
            user = authenticate(username=username, password=raw_password)
            user.first_name = firstname
            user.last_name = lastname
            user.save()
            msg = _('User created - please <a href="/login">login</a>.')
            success = True

            # return redirect("/login/")

        else:
            msg = _('Form is not valid')
    else:
        form = SignUpForm()

    return render(request, "accounts/register.html", {"form": form, "msg": msg, "success": success})

@login_required(login_url="/login/")
def edit_user(request):
    user = request.user
    executed_experiments = CaseStudy.objects.filter(user=user, executed=100).count()
    total_experiments = CaseStudy.objects.filter(user=user).count()
    if request.method == 'POST':
        form = UserForm(request.POST, instance=user)
        if form.is_valid():
            form.save()
            return HttpResponseRedirect(reverse("home"))
    else:
        form = UserForm(instance=user)
    return render(request, 'home/profile.html', {'form': form, 'executed_experiments': executed_experiments, 'total_experiments': total_experiments})
