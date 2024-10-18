from django.contrib import admin
from .models import UIElementsClassification, UIElementsDetection, FeatureExtractionTechnique, Prefilters, Postfilters, CNNModels, Postprocessing

# Register your models here.
admin.site.register(Prefilters)
admin.site.register(Postfilters)
admin.site.register(CNNModels)
admin.site.register(UIElementsDetection)
admin.site.register(UIElementsClassification)
admin.site.register(FeatureExtractionTechnique)
admin.site.register(Postprocessing)