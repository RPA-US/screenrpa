from django.urls import path
from . import views

urlpatterns = [
    path('feature-extraction-technique/list/', views.FeatureExtractionTechniqueListView.as_view(), name='fe_technique_list'),
    path('feature-extraction-technique/new/', views.FeatureExtractionTechniqueCreateView.as_view(), name='fe_technique_create'),
    path('ui-element-detection/list/', views.UIElementsDetectionListView.as_view(), name='ui_detection_list'),
    path('ui-element-detection/new/', views.UIElementsDetectionCreateView.as_view(), name='ui_detection_create'),
    path('noise-filtering/list/', views.NoiseFilteringListView.as_view(), name='noise_filtering_list'),
    path('noise-filtering/new/', views.NoiseFilteringCreateView.as_view(), name='noise_filtering_create'),
    path('ui-element-classification/list/', views.UIElementsClassificationListView.as_view(), name='ui_classification_list'),
    path('ui-element-classification/new/', views.UIElementsClassificationCreateView.as_view(), name='ui_classification_create'),
    # path('classify-image-components/', views.ui_elements_classification.as_view(), name='classify-image-components'),
    # path('detect-images-components/', views.detect_images_components.as_view(), name='detect-images-components'),
    # path('ocr/', views.get_ocr_image.as_view(),name='ocr'),
    # path('pad/', views.pad.as_view(), name='pad'), 
]