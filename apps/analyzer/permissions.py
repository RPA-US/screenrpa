"""
Possible functions for the ``PRIVATE_STORAGE_AUTH_FUNCTION`` setting.
"""
from .models import CaseStudy

def allow_staff(private_file):
    resource_name = private_file.relative_name[:len(private_file.relative_name)]
    foldername = resource_name.split("unzipped/")[1].split('/')[0]
    return CaseStudy.objects.filter(user=private_file.request.user, exp_foldername=foldername).exists()