from django.db import models
from django.urls import reverse
from django.contrib.auth.models import User
from private_storage.fields import PrivateFileField

# Create your models here.
class PDD(models.Model):
    created_at = models.DateTimeField(auto_now_add=True)
    file = PrivateFileField("PDD")
    case_study = models.ForeignKey('apps_analyzer.CaseStudy', on_delete=models.CASCADE, null=True) 
    user = models.ForeignKey(User, on_delete=models.CASCADE)

    class Meta:
        verbose_name = ("PDD")
        verbose_name_plural = ("PDDs")

    def __str__(self):
        return self.name

    # def get_absolute_url(self):
    #     return reverse("PDD_detail", kwargs={"pk": self.pk})
    
    def get_absolute_url(self):
        return reverse("reporting:report_list", args=[str(self.case_study_id)])
