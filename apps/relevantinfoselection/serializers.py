from rest_framework import serializers
from .models import GazeAnalysis
       
class GazeAnalysisSerializer(serializers.ModelSerializer):
    class Meta:
        model = GazeAnalysis
        fields = '__all__'