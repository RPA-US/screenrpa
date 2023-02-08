from django.db import models

# Create your models here.
from email.policy import default
from xmlrpc.client import Boolean
from django.db import models
from django.contrib.postgres.fields import ArrayField

# Create your models here.

def get_default_extract_training_columns_to_ignore():
    return 'Coor_X, Coor_Y, Case'.split(', ') # this returns a list

def get_default_decision_tree_columns_to_ignore():
    return 'Timestamp_start, Timestamp_end'.split(', ') # this returns a list

def get_default_algorithms():
    return 'ID3, CART, CHAID, C4.5'.split(', ') # this returns a list

class ExtractTrainingDataset(models.Model):
    columns_to_drop = ArrayField(models.CharField(max_length=25), default=get_default_extract_training_columns_to_ignore)
    columns_to_drop_before_decision_point = ArrayField(models.CharField(max_length=25), default=get_default_extract_training_columns_to_ignore)
    
class DecisionTreeTraining(models.Model):
    library = models.CharField(max_length=255, default='chefboost') # 'sklearn'
    algorithms = ArrayField(models.CharField(max_length=25), default=get_default_algorithms)
    one_hot_columns = ArrayField(models.CharField(max_length=25), default=get_default_algorithms)
    columns_to_drop = ArrayField(models.CharField(max_length=50), default=get_default_decision_tree_columns_to_ignore)
    columns_to_drop_before_decision_point = ArrayField(models.CharField(max_length=50), default=get_default_decision_tree_columns_to_ignore)