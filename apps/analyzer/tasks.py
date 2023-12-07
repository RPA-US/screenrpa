from celery import shared_task
import apps.analyzer.views as analyzer

# Functions in this file with the shared_task decorator will be picked up by celery and executed asynchronously

@shared_task()
def celery_task_process_case_study(case_study_id, phase):
    analyzer.case_study_generator_execution(case_study_id, phase)
