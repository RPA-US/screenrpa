from .models import ProcessDiscovery
from django.shortcuts import get_object_or_404

###########################################################################################################################
# case study get phases data  ###########################################################################################
###########################################################################################################################

def get_process_discovery(case_study):
  return get_object_or_404(ProcessDiscovery, case_study=case_study)

def case_study_has_process_discovery(case_study):
  return ProcessDiscovery.objects.filter(case_study=case_study, active=True).exists()
