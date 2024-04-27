from django.db import models
from django.urls import reverse
from django.contrib.auth.models import User
from private_storage.fields import PrivateFileField
from django.contrib.postgres.fields import ArrayField, JSONField
from xmlrpc.client import Boolean

# Create your models here.
class PDD(models.Model):
    created_at = models.DateTimeField(auto_now_add=True)
    file = PrivateFileField("PDD")
    execution = models.ForeignKey('apps_analyzer.Execution', on_delete=models.CASCADE, null=True) 
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    objective = models.CharField(max_length=255)
    purpose = models.CharField(max_length=255)
    process_overview = BooleanField(default=False)
    applications_used = BooleanField(default=False)
    as_is_process_map = BooleanField(default=False)
    detailed_as_is_process_actions = BooleanField(default=False)
    input_data_descrption = BooleanField(default=False)



    class Meta:
        verbose_name = ("PDD")
        verbose_name_plural = ("PDDs")

    def __str__(self):
        return self.name

    # def get_absolute_url(self):
    #     return reverse("PDD_detail", kwargs={"pk": self.pk})
    
    def get_absolute_url(self):
        return reverse("reporting:report_list", args=[str(self.case_study_id)])
