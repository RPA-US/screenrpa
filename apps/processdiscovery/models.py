from django.db import models
from django.db.models import JSONField
from django.contrib.auth.models import User
from django.urls import reverse

def default_process_discovery():
    return dict({"model_type": "",
                 "model_weights": "",
                 "clustering_type": "",
                 "labeling": "automatic",
                 "use_pca": False,
                 "n_components": 2,
                 "show_dendrogram:": False,
                 })


class ProcessDiscovery(models.Model):
    created_at = models.DateTimeField(auto_now_add=True)
    active = models.BooleanField(default=False, editable=True)
    executed = models.IntegerField(default=0, editable=True)
    type = models.CharField(max_length=25, default='rpa-us')
    configurations = JSONField(null=True, blank=True, default=default_process_discovery)
    skip = models.BooleanField(default=False)
    case_study = models.ForeignKey('apps_analyzer.CaseStudy', on_delete=models.CASCADE, null=True) 
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    
    def get_absolute_url(self):
        return reverse("processdiscovery:processdiscovery_list", args=[str(self.case_study_id)])
    
    def __str__(self):
        return 'type: ' + self.type