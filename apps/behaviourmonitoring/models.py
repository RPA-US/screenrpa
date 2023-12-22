from django.db import models
from django.contrib.auth.models import User
from django.db.models import JSONField
from django.urls import reverse
from django.core.exceptions import ValidationError

def default_monitoring_conf():
    return dict({
                "format": "mht_csv",
                "org:resource": "User1",
                "mht_log_filename": "Recording_20230424_1222.mht",
                "eyetracking_log_filename": "ET_RExtAPI-GazeAnalysis.csv",
                "native_slide_events": "Native_SlideEvents.csv",
                "ui_log_adjustment": "0.",
                "gaze_log_adjustment": "0.",
                "separator": ","
            })

# Create your models here.
class Monitoring(models.Model):
    created_at = models.DateTimeField(auto_now_add=True)
    type = models.CharField(max_length=25, default='imotions')
    executed = models.IntegerField(default=0, editable=True)
    active = models.BooleanField(default=False, editable=True)
    freeze = models.BooleanField(default=False, editable=True)
    ub_log_path = models.CharField(max_length=250, blank=True, null=True, default=None)
    configurations = JSONField(null=True, blank=True, default=default_monitoring_conf)
    case_study = models.ForeignKey('apps_analyzer.CaseStudy', on_delete=models.CASCADE, null=True) 
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    
    def clean(self):
        monitorings = Monitoring.objects.filter(case_study=self.case_study_id, active=True).exclude(id=self.id)
        # If there is more than one active monitoring, raise an error
        if self.active and len(monitorings) > 0:
            raise ValidationError('There is already an active monitoring for this case study.')
            
    def save(self, *args, **kwargs):
        self.full_clean()
        super(Monitoring, self).save(*args, **kwargs)
    
    def get_absolute_url(self):
        return reverse("behaviourmonitoring:monitoring_list", args=[str(self.case_study_id)])
    
    def __str__(self):
        return 'type: ' + self.type
