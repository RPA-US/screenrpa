from rest_framework import serializers
from .models import UIElementsClassification, FeatureExtractionTechnique, UIElementsDetection, NoiseFiltering

class UIElementsDetectionSerializer(serializers.ModelSerializer):
    class Meta:
        model = UIElementsDetection
        fields = '__all__' 
        
class NoiseFilteringSerializer(serializers.ModelSerializer):
    class Meta:
        model = NoiseFiltering
        fields = '__all__'

class UIElementsClassificationSerializer(serializers.ModelSerializer):
    class Meta:
        model = UIElementsClassification
        fields = '__all__'

class FeatureExtractionTechniqueSerializer(serializers.ModelSerializer):
    class Meta:
        model = FeatureExtractionTechnique
        fields = '__all__'