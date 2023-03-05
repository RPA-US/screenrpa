from django.urls import path
from . import views

urlpatterns = [
    path('extract-training-dataset/list/', views.ExtractTrainingDatasetListView.as_view(), name='extract_training_dataset_list'),
    path('extract-training-dataset/new/', views.ExtractTrainingDatasetCreateView.as_view(), name='extract_training_dataset_create'),
    path('decision-tree-training/list/', views.DecisionTreeTrainingListView.as_view(), name='decision_tree_training_list'),
    path('decision-tree-training/new/', views.DecisionTreeTrainingCreateView.as_view(), name='decision_tree_training_create'),
    # path('flat-dataset-row/', views.flat_dataset_row.as_view(), name='flat-dataset-row'),
    # path('plot-decision-tree/', views.plot_decision_tree.as_view(),name='plot-decision-tree'),
]