from rest_framework import serializers
from .models import Prefilters, UIElementsDetection, UIElementsClassification, FeatureExtractionTechnique, Filters

class PrefiltersSerializer(serializers.ModelSerializer):
    class Meta:
        model = Prefilters
        fields = '__all__'

class UIElementsDetectionSerializer(serializers.ModelSerializer):
    class Meta:
        model = UIElementsDetection
        fields = '__all__' 

class UIElementsClassificationSerializer(serializers.ModelSerializer):
    class Meta:
        model = UIElementsClassification
        fields = '__all__'
        
class FiltersSerializer(serializers.ModelSerializer):
    class Meta:
        model = Filters
        fields = '__all__'

class FeatureExtractionTechniqueSerializer(serializers.ModelSerializer):
    class Meta:
        model = FeatureExtractionTechnique
        fields = '__all__'