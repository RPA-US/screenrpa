from django.contrib import admin
from .models import CaseStudy, NoiseFiltering, UIElementsDetection, UIElementsClassification, FeatureExtractionTechnique, ExtractTrainingDataset, DecisionTreeTraining

# Register your models here.
admin.site.register(CaseStudy)
admin.site.register(NoiseFiltering)
admin.site.register(UIElementsDetection)
admin.site.register(UIElementsClassification)
admin.site.register(FeatureExtractionTechnique)
admin.site.register(ExtractTrainingDataset)
admin.site.register(DecisionTreeTraining)