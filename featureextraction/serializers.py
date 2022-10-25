from rest_framework import serializers
from .models import UIElementsClassification, FeatureExtractionTechnique, UIElementsDetection

class UIElementsDetectionSerializer(serializers.ModelSerializer):
    class Meta:
        model = UIElementsDetection
        fields = '__all__' # ['eyetracking_log_filename', 'add_words_columns', 'overwrite_info', 'algorithm']

class UIElementsClassificationSerializer(serializers.ModelSerializer):
    class Meta:
        model = UIElementsClassification
        fields = '__all__'

class FeatureExtractionTechniqueSerializer(serializers.ModelSerializer):
    class Meta:
        model = FeatureExtractionTechnique
        fields = '__all__'