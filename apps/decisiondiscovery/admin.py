from django.contrib import admin
from .models import ExtractTrainingDataset, DecisionTreeTraining

# Register your models here.
admin.site.register(ExtractTrainingDataset)
admin.site.register(DecisionTreeTraining)