from rest_framework import serializers
from .models import CaseStudy

class CaseStudySerializer(serializers.ModelSerializer):
    # phases_to_execute = serializers.JSONField()
    special_colnames = serializers.JSONField()
    class Meta:
        model = CaseStudy
        fields = '__all__' # ['id', 'title', 'created_at', 'exp_version_name', 'phases_to_execute', 'decision_point_activity', 'path_to_save_experiment', 'gui_class_success_regex', 'gui_quantity_difference', 'scenarios_to_study', 'drop', 'user']
    
    # def create(self, validated_data):
    #     validated_data.pop('phases_to_execute')
    #     return CaseStudy.objects.create(**validated_data)