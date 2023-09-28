from django.db import models
from django.contrib.auth.models import User
from django.db.models import JSONField
from django.urls import reverse
from apps.analyzer.models import CaseStudy


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
    configurations = JSONField(null=True, blank=True, default=default_monitoring_conf)
    case_study = models.ForeignKey(CaseStudy, on_delete=models.CASCADE, null=True) 
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    
    def get_absolute_url(self):
        return reverse("behaviourmonitoring:monitoring_list")
    
    def __str__(self):
        return 'type: ' + self.type
