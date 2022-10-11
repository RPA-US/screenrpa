from rest_framework import serializers
from .models import CaseStudy, ClassifyImageComponents, DecisionTreeTraining, FeatureExtractionTechnique, ExtractTrainingDataset, GUIComponentDetection

class GUIComponentDetectionSerializer(serializers.ModelSerializer):
    class Meta:
        model = GUIComponentDetection
        fields = '__all__' # ['eyetracking_log_filename', 'add_words_columns', 'overwrite_npy', 'algorithm']

class ClassifyImageComponentsSerializer(serializers.ModelSerializer):
    class Meta:
        model = ClassifyImageComponents
        fields = '__all__' # ['model_json_file_name', 'model_weights', 'model_properties', 'algorithm']

class FeatureExtractionTechniqueSerializer(serializers.ModelSerializer):
    class Meta:
        model = FeatureExtractionTechnique
        fields = '__all__'

class ExtractTrainingDatasetSerializer(serializers.ModelSerializer):
    class Meta:
        model = ExtractTrainingDataset
        fields = '__all__' # ['columns_to_ignore']

    def create(self, validated_data):
        return ExtractTrainingDataset.objects.create(**validated_data)
    
class DecisionTreeTrainingSerializer(serializers.ModelSerializer):
    class Meta:
        model = DecisionTreeTraining
        fields = '__all__' # ['library', 'algorithms', 'mode', 'columns_to_ignore']

    def create(self, validated_data):
        return DecisionTreeTraining.objects.create(**validated_data)

class CaseStudySerializer(serializers.ModelSerializer):
    phases_to_execute = serializers.JSONField()
    special_colnames = serializers.JSONField()
    class Meta:
        model = CaseStudy
        fields = '__all__' # ['id', 'title', 'created_at', 'exp_version_name', 'phases_to_execute', 'decision_point_activity', 'path_to_save_experiment', 'gui_class_success_regex', 'gui_quantity_difference', 'scenarios_to_study', 'drop', 'user']
    
    def create(self, validated_data):
        validated_data.pop('phases_to_execute')
        return CaseStudy.objects.create(**validated_data)