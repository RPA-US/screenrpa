from .models import Monitoring
from django.shortcuts import get_object_or_404

###########################################################################################################################
# case study get phases data  ###########################################################################################
###########################################################################################################################

def get_monitoring_from_cs(case_study):
  return get_object_or_404(Monitoring, case_study=case_study, active=True)

def case_study_has_monitoring_from_cs(case_study):
  return Monitoring.objects.filter(case_study=case_study, active=True).exists()