from celery import shared_task
from analyzer.serializers import CaseStudySerializer
import analyzer.views as analyzer
from rest_framework import status

# Functions in this file with the shared_task decorator will be picked up by celery and executed asynchronously

@shared_task()
def init_case_study_task(request_data, *args, **kwargs):
    case_study_serialized = CaseStudySerializer(data=request_data)
    st = status.HTTP_200_OK

    if not case_study_serialized.is_valid():
        response_content = case_study_serialized.errors
        st=status.HTTP_400_BAD_REQUEST
    else:
        execute_case_study = True
        try:
            # if not (case_study_serialized.data['mode'] in ['generation', 'results', 'both']):
            #     response_content = {"message": "mode must be one of the following options: generation, results, both."}
            #     st = status.HTTP_422_UNPROCESSABLE_ENTITY
            #     execute_case_study = False
            #     return Response(response_content, status=st)

            if not isinstance(case_study_serialized.data['phases_to_execute'], dict):
                response_content = {"message": "phases_to_execute must be of type dict!!!!! and must be composed by phases contained in ['ui_elements_detection','ui_elements_classification','feature_extraction','extract_training_dataset','decision_tree_training']"}
                st = status.HTTP_422_UNPROCESSABLE_ENTITY
                execute_case_study = False
                return (response_content, st)

            if not case_study_serialized.data['phases_to_execute']['ui_elements_detection']['type'] in ["rpa-us", "uied"]:
                response_content = {"message": "Elements Detection type must be one of ['rpa-us', 'uied']"}
                st = status.HTTP_422_UNPROCESSABLE_ENTITY
                execute_case_study = False
                return (response_content, st)

            if not case_study_serialized.data['phases_to_execute']['ui_elements_classification']['type'] in ["rpa-us", "uied"]:
                response_content = {"message": "Elements Classification type must be one of ['rpa-us', 'uied']"}
                st = status.HTTP_422_UNPROCESSABLE_ENTITY
                execute_case_study = False
                return (response_content, st)

            for phase in dict(case_study_serialized.data['phases_to_execute']).keys():
                if not(phase in ['ui_elements_detection','ui_elements_classification','feature_extraction','extract_training_dataset','decision_tree_training']):
                    response_content = {"message": "phases_to_execute must be composed by phases contained in ['ui_elements_detection','ui_elements_classification','feature_extraction','extract_training_dataset','decision_tree_training']"}
                    st = status.HTTP_422_UNPROCESSABLE_ENTITY
                    execute_case_study = False
                    return (response_content, st)

            if execute_case_study:
                generator_success, case_study = analyzer.case_study_generator(case_study_serialized.data)
                response_content = {"message": "Case Study generated"}
                if not generator_success:
                    st = status.HTTP_422_UNPROCESSABLE_ENTITY
                    response_content = {"message": "Case Study generation failed"}

        except Exception as e:
            response_content = {"message": "Some of atributes are invalid: " + str(e) }
            st = status.HTTP_422_UNPROCESSABLE_ENTITY

    # item = CaseStudy.objects.create(serializer)
    # result = CaseStudySerializer(item)
    # return Response(result.data, status=status.HTTP_201_CREATED)

    return (response_content, st)