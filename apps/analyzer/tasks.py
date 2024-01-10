from celery import shared_task
import apps.analyzer.views as analyzer
from apps.analyzer.models import Execution

# Functions in this file with the shared_task decorator will be picked up by celery and executed asynchronously

@shared_task()
def celery_task_process_case_study(execution_id):
    execution = Execution.objects.get(id=execution_id)
    analyzer.case_study_generator_execution(execution)
