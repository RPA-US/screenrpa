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
    objective = models.CharField(max_length=255, null=True, blank=True)
    purpose = models.CharField(max_length=255, null=True, blank=True)
    process_overview = models.BooleanField(editable=True, default=True)
    applications_used = models.BooleanField(default=False)
    as_is_process_map = models.BooleanField(default=False)
    detailed_as_is_process_actions = models.BooleanField(default=False)
    input_data_description = models.BooleanField(default=False)



    class Meta:
        verbose_name = ("PDD")
        verbose_name_plural = ("PDDs")

    def __str__(self):
        return self.name

    # def get_absolute_url(self):
    #     return reverse("PDD_detail", kwargs={"pk": self.pk})
    
    def get_absolute_url(self):
        return reverse("analyzer:execution_detail", kwargs={"execution_id": self.execution.id})
