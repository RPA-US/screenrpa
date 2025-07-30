from django.db import models
from django.contrib.auth.models import User
from django.urls import reverse
from django.core.exceptions import ValidationError
from private_storage.fields import PrivateFileField
from apps.analyzer.models import CaseStudy, Execution
import os
from django.utils.translation import gettext_lazy as _

class FitbitToken(models.Model):
    user = models.OneToOneField(User, on_delete=models.CASCADE)
    access_token = models.TextField()
    refresh_token = models.TextField()
    expires_in = models.IntegerField()
    token_type = models.CharField(max_length=20)
    scope = models.CharField(max_length=200)
    fitbit_user_id = models.CharField(max_length=100)
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"Token for user {self.user_id}"
    
class BiometricAnalysisConfig(models.Model):
    """
    Configuración para el análisis biométrico de un caso de estudio.
    Similar a otras fases como Monitoring, ProcessDiscovery, etc.
    """
    title = models.CharField(max_length=255, default="Análisis Biométrico")
    description = models.TextField(blank=True, null=True)
    
    # Configuración general
    active = models.BooleanField(default=False, verbose_name=_("Activo"))
    created_at = models.DateTimeField(auto_now_add=True)
    executed = models.IntegerField(default=0, editable=True)
    freeze = models.BooleanField(default=False, editable=True)
    
    # Relaciones
    case_study = models.ForeignKey(CaseStudy, on_delete=models.CASCADE, related_name='biometric_configs')
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    
    # Configuración específica
    METRIC_CHOICES = [
        ('fc', 'Frecuencia Cardíaca'),
        ('pasos', 'Pasos'),
        ('calorias', 'Calorías'),
        ('zona_activa', 'Zona Activa'),
        ('sedentario', 'Tiempo Sedentario'),
        ('ratio_fc_pasos', 'Ratio FC/Pasos'),
        ('cvl', 'CVL'),
        ('sdnn', 'SDNN'),
    ]
    
    CHART_TYPE_CHOICES = [
    ('line', 'Gráfico de Líneas'),
    ('bar', 'Gráfico de Barras'),
    ('radar', 'Gráfico Radar'),
    ]
    
    default_metrics = models.JSONField(default=list, verbose_name=_("Métricas por defecto"))
    default_chart_type = models.CharField(max_length=20, choices=CHART_TYPE_CHOICES, default='line', verbose_name=_("Tipo de gráfico por defecto"))
    
    def clean(self):
        """Verifica que solo haya una configuración activa por caso de estudio"""
        configs = BiometricAnalysisConfig.objects.filter(
            case_study=self.case_study, 
            active=True
        ).exclude(id=self.id)
        
        if self.active and configs.exists():
            raise ValidationError(_('Ya existe una configuración activa para este caso de estudio.'))
    
    def save(self, *args, **kwargs):
        # Si se activa esta configuración, desactiva las demás del mismo caso de estudio
        if self.active:
            BiometricAnalysisConfig.objects.filter(
                case_study=self.case_study,
                active=True
            ).exclude(pk=self.pk).update(active=False)
        self.full_clean()
        super().save(*args, **kwargs)
    
    def get_absolute_url(self):
        return reverse("wearabletracking:biometric_config_detail", kwargs={'pk': self.pk})
    
    def __str__(self):
        return f"Configuración de análisis biométrico: {self.title}"
    
    class Meta:
        verbose_name = _("Configuración de análisis biométrico")
        verbose_name_plural = _("Configuraciones de análisis biométrico")

class BiometricAnalysisReport(models.Model):
    """
    Representa un reporte de análisis biométrico generado durante una ejecución.
    """
    title = models.CharField(max_length=255)
    created_at = models.DateTimeField(auto_now_add=True)
    
    # Relaciones
    execution = models.OneToOneField(Execution, on_delete=models.CASCADE, related_name='biometric_report')
    config = models.ForeignKey(BiometricAnalysisConfig, on_delete=models.CASCADE, related_name='reports')
    
    # Datos del reporte
    merged_file = models.CharField(max_length=255, default='merged_ui_wearable.csv')
    metrics = models.JSONField(default=list, verbose_name=_("Métricas seleccionadas"))
    chart_type = models.CharField(max_length=20, choices=BiometricAnalysisConfig.CHART_TYPE_CHOICES, default='line')
    
    # Archivo del reporte generado
    report_file = PrivateFileField("Reporte PDF", upload_to='biometric_reports/', null=True, blank=True)

    extra_data = models.JSONField(default=dict, blank=True)
    
    def get_merged_file_path(self):
        """Retorna la ruta completa al archivo CSV combinado"""
        if self.execution:
            return os.path.join(self.execution.exp_folder_complete_path, self.merged_file)
        return None
    
    def get_absolute_url(self):
        return reverse("wearabletracking:biometric_report_detail", kwargs={'pk': self.pk})
    
    def __str__(self):
        return f"Reporte: {self.title} ({self.created_at.strftime('%Y-%m-%d')})"
    
    class Meta:
        verbose_name = _("Reporte de análisis biométrico")
        verbose_name_plural = _("Reportes de análisis biométrico")