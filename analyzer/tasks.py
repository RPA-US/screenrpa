from celery import shared_task
import analyzer.views as analyzer

# Functions in this file with the shared_task decorator will be picked up by celery and executed asynchronously

@shared_task()
def init_generate_case_study(case_study_id):
    analyzer.generate_case_study(case_study_id)
