from rest_framework import serializers
from .models import Preselectors, UIElementsDetection, UIElementsClassification, FeatureExtractionTechnique, Selectors

class PreselectorsSerializer(serializers.ModelSerializer):
    class Meta:
        model = Preselectors
        fields = '__all__'

class UIElementsDetectionSerializer(serializers.ModelSerializer):
    class Meta:
        model = UIElementsDetection
        fields = '__all__' 

class UIElementsClassificationSerializer(serializers.ModelSerializer):
    class Meta:
        model = UIElementsClassification
        fields = '__all__'
        
class SelectorsSerializer(serializers.ModelSerializer):
    class Meta:
        model = Selectors
        fields = '__all__'

class FeatureExtractionTechniqueSerializer(serializers.ModelSerializer):
    class Meta:
        model = FeatureExtractionTechnique
        fields = '__all__'