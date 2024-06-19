import json
from django.shortcuts import get_object_or_404, render
from django.views import View
from django.contrib.auth.mixins import LoginRequiredMixin
from django.contrib.auth.decorators import login_required
from django.http import HttpResponse
from .models import Notification

# Create your views here.

@login_required(login_url='/login/')
def get_notifications(request):
    queryset = Notification.objects.filter(user=request.user).order_by('-created_at')
    res = {
        'unread': Notification.objects.filter(user=request.user, read=False).count(),
        'notifications': queryset.get() if queryset.exists() else []
    }
    return HttpResponse(json.dumps(res), content_type='application/json')

@login_required(login_url='/login/')
def mark_as_read(request):
    notification_id = request.POST.get('notification_id')
    notification = get_object_or_404(Notification, id=notification_id)
    if notification.user != request.user:
        return HttpResponse(status=403)
    notification.read = True
    notification.save()
    return HttpResponse(json.dumps({'success': True}), content_type='application/json')

@login_required(login_url='/login/')
def delete_notification(request):
    notification_id = request.POST.get('notification_id')
    notification = get_object_or_404(Notification, id=notification_id)
    if notification.user != request.user:
        return HttpResponse(status=403)
    notification.delete()
    return HttpResponse(json.dumps({'success': True}), content_type='application/json')

## Utils for other modules

def create_notification(user, short, message, href):
    Notification.objects.create(user=user, short=short, message=message, href=href)