from django.urls import path
from . import views

app_name = 'featureextraction'

urlpatterns = [
    # UI Element Detection & Classification
    path('ui-element-detection/list/<int:case_study_id>/', views.UIElementsDetectionListView.as_view(), name='ui_detection_list'),
    path('ui-element-detection/new/<int:case_study_id>/', views.UIElementsDetectionCreateView.as_view(), name='ui_detection_create'),
    path('ui-element-detection/detail/<int:case_study_id>/<int:ui_elements_detection_id>/', views.UIElementsDetectionDetailView.as_view(), name='ui_element_detection_detail'),
    path('ex/ui-element-detection/detail/<int:execution_id>/<int:ui_elements_detection_id>/', views.UIElementsDetectionDetailView.as_view(), name='ui_element_detection_detail-execution'),
    
    path('ui-element-detection/active/', views.set_as_ui_elements_detection_active, name='ui_element_detection_set_as_active'),
    path('ui-element-detection/inactive/', views.set_as_ui_elements_detection_inactive, name='ui_element_detection_set_as_inactive'),
    path('ui-element-detection/delete/', views.delete_ui_elements_detection, name='ui_element_detection_delete'),
    path('ex/ui-element-detection/result/<int:execution_id>/som', views.UIElementsDetectionResultsGetImgSom.as_view(), name='uielmdet_result_som'),
    path('ex/ui-element-detection/result/<int:execution_id>/', views.UIElementsDetectionResultDetailView.as_view(), name='uielmdet_result'),
    ##
    # Pre-Filtering
    path('prefiltering/list/<int:case_study_id>/', views.PrefiltersListView.as_view(), name='prefilters_list'),
    path('prefiltering/new/<int:case_study_id>/', views.PrefiltersCreateView.as_view(), name='prefilters_create'),
    path('prefiltering/detail/<int:case_study_id>/<int:prefilter_id>/', views.PrefiltersDetailView.as_view(), name='prefilters_detail'),
    path('ex/prefiltering/detail/<int:execution_id>/<int:prefilter_id>/', views.PrefiltersDetailView.as_view(), name='prefilters_detail-execution'),
    path('ex/prefiltering/result/<int:execution_id>/', views.PrefilteringResultDetailView.as_view(), name='prefiltering_result'),

    path('prefiltering/active/', views.set_as_prefilters_active, name='prefilters_set_as_active'),
    path('prefiltering/inactive/', views.set_as_prefilters_inactive, name='prefilters_set_as_inactive'),
    path('prefiltering/delete/', views.delete_prefilter, name='prefilters_delete'),
    # path('prefiltering/list/', views.PrefiltersListView.as_view(), name='prefilters_list'),
    ##
    # Post-Filtering
    path('postfiltering/list/<int:case_study_id>/', views.PostfiltersListView.as_view(), name='postfilters_list'),
    path('postfiltering/new/<int:case_study_id>/', views.PostfiltersCreateView.as_view(), name='postfilters_create'),
    path('postfiltering/detail/<int:case_study_id>/<int:postfilter_id>/', views.PostfiltersDetailView.as_view(), name='prefilters_detail'),
    path('ex/postfiltering/detail/<int:execution_id>/<int:postfilter_id>/', views.PostfiltersDetailView.as_view(), name='prefilters_detail-execution'),
    path('ex/postfiltering/result/<int:execution_id>/', views.PostfilteringResultDetailView.as_view(), name='prefiltering_result'),

    path('postfiltering/active/', views.set_as_postfilters_active, name='postfilters_set_as_active'),
    path('postfiltering/inactive/', views.set_as_postfilters_inactive, name='postfilters_set_as_inactive'),
    path('postfiltering/delete/', views.delete_postfilter, name='postfilters_delete'),
    path('postfiltering/draw/<int:execution_id>/', views.draw_postfilter, name='postfilters_draw'),
    path('draw/<int:execution_id>/', views.draw_ui_compos, name='ui_compos_draw'),
    # path('postfiltering/list/', views.PostfiltersListView.as_view(), name='postfilters_list'),
    ##
    # Feature Extraction Technique
    path('feature-extraction-technique/list/<int:case_study_id>/', views.FeatureExtractionTechniqueListView.as_view(), name='fe_technique_list'),
    path('feature-extraction-technique/new/<int:case_study_id>/', views.FeatureExtractionTechniqueCreateView.as_view(), name='fe_technique_create'),
    path('feature-extraction-technique/detail/<int:case_study_id>/<int:feature_extraction_technique_id>/', views.FeatureExtractionTechniqueDetailView.as_view(), name='fe_technique_detail'),
    path('ex/feature-extraction-technique/detail/<int:execution_id>/<int:feature_extraction_technique_id>/', views.FeatureExtractionTechniqueDetailView.as_view(), name='fe_technique_detail-execution'),
    path('feature-extraction-technique/result/<int:execution_id>/', views.FeatureExtractionResultDetailView.as_view(), name='featureExtraction_result'),


    path('feature-extraction-technique/active/', views.set_as_feature_extraction_technique_active, name='fe_technique_set_as_active'),
    path('feature-extraction-technique/inactive/', views.set_as_feature_extraction_technique_inactive, name='fe_technique_set_as_inactive'),
    path('feature-extraction-technique/delete/', views.delete_feature_extraction_technique, name='fe_technique_delete'),
    # path('classify-image-components/', views.ui_elements_classification.as_view(), name='classify-image-components'),
    # path('detect-images-components/', views.detect_images_components.as_view(), name='detect-images-components'),
    # path('ocr/', views.get_ocr_image.as_view(),name='ocr'),
    # path('pad/', views.pad.as_view(), name='pad'), 

    ##
    # Post-Processing Techniques
    path('postprocessing/list/<int:case_study_id>/', views.PostprocessingListView.as_view(), name='postprocessing_list'),
    path('postprocessing/new/<int:case_study_id>/', views.PostprocessingCreateView.as_view(), name='postprocessing_create'),
    path('postprocessing/detail/<int:case_study_id>/<int:postprocessing_id>/', views.PostprocessingDetailView.as_view(), name='postprocessing_detail'),
    path('ex/postprocessing/detail/<int:execution_id>/<int:postprocessing_id>/', views.PostprocessingDetailView.as_view(), name='postprocessing-execution'),

    path('postprocessing/active/', views.set_as_postprocessing_active, name='postprocessing_set_as_active'),
    path('postprocessing/inactive/', views.set_as_postprocessing_inactive, name='postprocessing_set_as_inactive'),
    path('postprocessing/delete/', views.delete_postprocessing, name='postprocessing_delete'),
]