from .models import ProcessDiscovery
from django.shortcuts import get_object_or_404

###########################################################################################################################
# case study get phases data  ###########################################################################################
###########################################################################################################################

def get_process_discovery_from_cs(case_study):
  return get_object_or_404(ProcessDiscovery, case_study=case_study)
