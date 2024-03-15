from django.db import models
from django.db.models import JSONField
from django.contrib.auth.models import User
from private_storage.fields import PrivateFileField
from django.urls import reverse

def default_process_discovery():
    return dict({'draw_dendogram': True, 'similarity_th': 0.2})


# Create your models here.
class ProcessDiscovery(models.Model):
    preloaded = models.BooleanField(default=False, editable=True)
    title = models.CharField(max_length=255)
    preloaded_file = PrivateFileField("File", null=True, blank=True)
    freeze = models.BooleanField(default=False, editable=True)
    created_at = models.DateTimeField(auto_now_add=True)
    active = models.BooleanField(default=False, editable=True)
    executed = models.IntegerField(default=0, editable=True)
    type = models.CharField(max_length=25, default='rpa-us', null=True, blank=True )
    configurations = JSONField(null=True, blank=True, default=default_process_discovery)
    skip = models.BooleanField(default=False)
    case_study = models.ForeignKey('apps_analyzer.CaseStudy', on_delete=models.CASCADE, null=True) 
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    
    def get_absolute_url(self):
        return reverse("processdiscovery:processdiscovery_list", args=[str(self.case_study_id)])
    
    def __str__(self):
        return 'type: ' + self.type