from rest_framework import serializers
from .models import ProcessDiscovery

class ProcessDiscoverySerializer(serializers.ModelSerializer):
    class Meta:
        model = ProcessDiscovery
        fields = '__all__'

    def create(self, validated_data):
        return ProcessDiscovery.objects.create(**validated_data)