import json
from django.utils import timezone
from django.shortcuts import get_object_or_404
from django.contrib.auth.decorators import login_required
from django.http import HttpResponse
from .models import Notification, Status

# Create your views here.

@login_required(login_url='/login/')
def get_notifications(request):
    queryset = Notification.objects.filter(user=request.user).order_by('-created_at')
    data = []
    if queryset.exists():
        time_diff = timezone.now() - queryset[0].created_at
        data = list(map(lambda x: {
            'id': x.id,
            'short': x.short,
            'message': x.message,
            'read': x.read,
            'href': x.href,
            'timeDiff': time_diff.total_seconds(),
            'status': x.status
        }, queryset))
    res = {
        'unread': Notification.objects.filter(user=request.user, read=False).count(),
        'notifications': data
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
def mark_all_as_read(request):
    Notification.objects.filter(user=request.user, read=False).update(read=True)
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

def create_notification(user, short, message, href, status=Status.INFO.value):
    Notification.objects.create(user=user, short=short, message=message, href=href, status=status).save()