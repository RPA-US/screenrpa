from django.db import models
from django.db.models import JSONField
from django.contrib.auth.models import User
from private_storage.fields import PrivateFileField
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

    # New fields
    model_type = models.CharField(max_length=10, choices=[('vgg', 'VGG'), ('clip', 'Clip')], default='vgg')
    text_weight = models.DecimalField(max_digits=5, decimal_places=2, default=0.5)
    image_weight = models.DecimalField(max_digits=5, decimal_places=2, default=0.5)
    clustering_type = models.CharField(max_length=20, choices=[('hierarchical', 'Hierarchical')], default='hierarchical')
    labeling = models.CharField(max_length=10, choices=[('automatic', 'Automatic'), ('manual', 'Manual')], default='automatic')
    use_pca = models.BooleanField(default=False)
    n_components = models.FloatField(default=0.95)
    show_dendrogram = models.BooleanField(default=False) 

    def get_absolute_url(self):
        return reverse("processdiscovery:processdiscovery_list", args=[str(self.case_study_id)])
    
    def __str__(self):
        return 'type: ' + self.type