from django.db import models
from django.urls import reverse
from private_storage.fields import PrivateFileField
from apps.analyzer.models import Execution
import os
from django.utils.translation import gettext_lazy as _

class EmotionAnalysisReport(models.Model):
    """
    Representa un reporte de análisis de emociones generado durante una ejecución.
    """
    title = models.CharField(max_length=255)
    created_at = models.DateTimeField(auto_now_add=True)
    
    # Relaciones
    execution = models.ForeignKey(Execution, on_delete=models.CASCADE, related_name='emotion_reports')
    scenario = models.CharField(max_length=255, default='default')
    
    # Datos del reporte
    emotions_file = models.CharField(max_length=255, default='registros_emociones.csv')
    merged_file = models.CharField(max_length=255, default='merged_emotions_wearable.csv', blank=True)
    
    # Archivo del reporte generado
    report_file = PrivateFileField("Reporte PDF", upload_to='emotion_reports/', null=True, blank=True)
    
    # Datos procesados del CSV
    extra_data = models.JSONField(default=dict, blank=True)
    has_merged_data = models.BooleanField(default=False)
    
    def get_emotions_file_path(self):
        """Retorna la ruta completa al archivo CSV de emociones"""
        if self.execution:
            return os.path.join(self.execution.exp_folder_complete_path, self.scenario, self.emotions_file)
        return None
    
    def get_merged_file_path(self):
        """Retorna la ruta completa al archivo CSV combinado con wearables"""
        if self.execution and self.merged_file:
            return os.path.join(self.execution.exp_folder_complete_path, self.scenario, self.merged_file)
        return None
    
    def get_absolute_url(self):
        return reverse("emotions:emotion_report_detail", kwargs={'pk': self.pk})
    
    def __str__(self):
        return f"Reporte de emociones: {self.title} - {self.scenario}"
    
    class Meta:
        verbose_name = _("Reporte de análisis de emociones")
        verbose_name_plural = _("Reportes de análisis de emociones")
        unique_together = ('execution', 'scenario')