from django.db import models
from django.contrib.auth.models import User
from django.db.models import JSONField
from django.urls import reverse
from django.core.exceptions import ValidationError
from private_storage.fields import PrivateFileField
from django.utils.translation import gettext_lazy as _
from django.db.models.signals import pre_delete
from django.dispatch import receiver

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
    preloaded = models.BooleanField(default=False, editable=True)
    preloaded_file = PrivateFileField("File", null=True, blank=True)
    freeze = models.BooleanField(default=False, editable=True)
    created_at = models.DateTimeField(auto_now_add=True)
    title = models.CharField(max_length=100, default="New Monitoring")
    type = models.CharField(max_length=25, default='already_processed',blank=True, null=True)
    executed = models.IntegerField(default=0, editable=True)
    active = models.BooleanField(default=False, editable=True)
    ub_log_path = models.CharField(max_length=250, blank=True, null=True, default=None)
    # TODO: What do we do with format
    format = models.CharField(max_length=25, default='mht_csv')
    ui_log_filename = models.CharField(max_length=100, default='Recording_20240617_1711.mht')
    ui_log_separator = models.CharField(max_length=1, default=',')
    gaze_log_filename = models.CharField(max_length=100, default='ET_RExtAPI-GazeAnalysis.csv')
    gaze_log_adjustment = models.FloatField(default=0)
    native_slide_events = models.CharField(max_length=100, default='Native_SlideEvents.csv')
    screen_inches = models.FloatField(null =True, blank = True ,default=15.6)
    observer_camera_distance = models.FloatField(null =True, blank = True ,default=50)
    screen_width = models.IntegerField(null =True, blank = True ,default=1920)
    screen_height = models.IntegerField(null =True, blank = True ,default=1080)

    case_study = models.ForeignKey('apps_analyzer.CaseStudy', on_delete=models.CASCADE, null=True) 
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    
    def clean(self):
        monitorings = Monitoring.objects.filter(case_study=self.case_study_id, active=True).exclude(id=self.id)
        # If there is more than one active monitoring, raise an error
        if self.active and len(monitorings) > 0:
            raise ValidationError(_('There is already an active monitoring for this case study.'))
            
    def save(self, *args, **kwargs):
        self.full_clean()
        super(Monitoring, self).save(*args, **kwargs)
    
    def get_absolute_url(self):
        return reverse("behaviourmonitoring:monitoring_list", args=[str(self.case_study_id)])
    
    def __str__(self):
        return 'type: ' + self.type

@receiver(pre_delete, sender=Monitoring)
def monitoring_delete(sender, instance, **kwargs):
    if instance.preloaded_file:
        instance.preloaded_file.delete(save=False)