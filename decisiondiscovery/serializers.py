from rest_framework import serializers
from .models import DecisionTreeTraining, ExtractTrainingDataset

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