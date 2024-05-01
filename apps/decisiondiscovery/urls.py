from django.urls import path
from . import views

app_name = 'decisiondiscovery'

urlpatterns = [
    # Extract Training Dataset
    path('extract-training-dataset/list/<int:case_study_id>/', views.ExtractTrainingDatasetListView.as_view(), name='extract_training_dataset_list'),
    path('extract-training-dataset/new/<int:case_study_id>/', views.ExtractTrainingDatasetCreateView.as_view(), name='extract_training_dataset_create'),
    path('extract-training-dataset/detail/<int:case_study_id>/<int:extract_training_dataset_id>/', views.ExtractTrainingDatasetDetailView.as_view(), name='extract_training_dataset_detail'),
    path('extract-training-dataset/active/', views.set_as_extracting_training_dataset_active, name='extract_training_dataset_set_as_active'),
    path('extract-training-dataset/inactive/', views.set_as_extracting_training_dataset_inactive, name='extract_training_dataset_set_as_inactive'),
    path('extract-training-dataset/delete/', views.delete_extracting_training_dataset, name='extract_training_dataset_delete'),
    path('extract-training-dataset/inactive/', views.set_as_extracting_training_dataset_inactive, name='extract_training_dataset_set_as_inactive'),
    ##
    # Decision Tree Training
    path('decision-tree-training/list/<int:case_study_id>/', views.DecisionTreeTrainingListView.as_view(), name='decision_tree_training_list'),
    path('decision-tree-training/new/<int:case_study_id>/', views.DecisionTreeTrainingCreateView.as_view(), name='decision_tree_training_create'),
    path('decision-tree-training/detail/<int:case_study_id>/<int:decision_tree_training_id>/', views.DecisionTreeTrainingDetailView.as_view(), name='decision_tree_training_detail'),
    path('ex/decision-tree-training/detail/<int:execution_id>/<int:decision_tree_training_id>/', views.DecisionTreeTrainingDetailView.as_view(), name='decision_tree_training_details-execution'),
    path('decision-tree-training/active/', views.set_as_decision_tree_training_active, name='decision_tree_training_set_as_active'),
    path('decision-tree-training/delete/', views.delete_decision_tree_training, name='decision_tree_training_delete'),
    path('decision-tree-training/inactive/', views.set_as_decision_tree_training_inactive, name='decision_tree_training_set_as_inactive'),
    # path('decision-tree-training/results/', views.decision_tree_feature_checker, name='dt_results'),
    # path('flat-dataset-row/', views.flat_dataset_row.as_view(), name='flat-dataset-row'),
    # path('plot-decision-tree/', views.plot_decision_tree.as_view(),name='plot-decision-tree'),
]