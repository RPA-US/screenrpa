"""
Possible functions for the ``PRIVATE_STORAGE_AUTH_FUNCTION`` setting.
"""
from .models import CaseStudy

def allow_staff(private_file):
    name = private_file.relative_name[:len(private_file.relative_name)-1]
    return CaseStudy.objects.filter(user=private_file.request.user, exp_file=name).exists()