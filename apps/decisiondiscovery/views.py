import os
from django.forms.models import model_to_dict
from django.http import HttpResponseRedirect
from django.views.generic import ListView, DetailView, CreateView
from django.views.generic.edit import FormMixin
from django.shortcuts import render, get_object_or_404
from django.urls import reverse
from django.core.exceptions import ValidationError
from tqdm import tqdm
from art import tprint
import pandas as pd
from apps.chefboost import Chefboost as chef
from apps.analyzer.models import CaseStudy 
from core.settings import sep, DECISION_FOLDERNAME, PLATFORM_NAME, FLATTENING_PHASE_NAME, DECISION_MODEL_DISCOVERY_PHASE_NAME, FLATTENED_DATASET_NAME
from core.utils import read_ui_log_as_dataframe
from .models import DecisionTreeTraining, ExtractTrainingDataset
from .forms import DecisionTreeTrainingForm, ExtractTrainingDatasetForm
from .decision_trees import sklearn_decision_tree, chefboost_decision_tree
from .flattening import flat_dataset_row
from .utils import find_path_in_decision_tree, parse_decision_tree
from django.utils.translation import gettext_lazy as _

# def clean_dataset(df):
#     assert isinstance(df, pd.DataFrame), "df needs to be a pd.DataFrame"
#     df.dropna(inplace=True)
#     indices_to_keep = ~df.isin([np.nan, np.inf, -np.inf]).any(1)
#     return df[indices_to_keep].astype(np.float64)

def extract_training_dataset(log_path, root_path, execution):
    """
    Iterate for every UI log row:
        For each case:
            Store in a map all the attributes of the activities until reaching the decision point
            Assuming the decision point is on activity D, the map would have the following structure:
        {
            "headers": ["timestamp", "MOUSE", "clipboard"...],
            "case1": {"CoorX_A": "value1", "CoorY_A": "value2", ..., "Checkbox_A: "value3",
                        "CoorX_B": "value1", "CoorY_B": "value2", ..., "Checkbox_B: "value3"
                        "CoorX_C": "value1", "CoorY_C": "value2", ..., "Checkbox_C: "value3"
                },
            ...
            
            "caseN": {"CoorX_A": "value1", "CoorY_A": "value2", ..., "Checkbox_A: "value3",
                        "CoorX_B": "value1", "CoorY_B": "value2", ..., "Checkbox_B: "value3"
                        "CoorX_C": "value1", "CoorY_C": "value2", ..., "Checkbox_C: "value3"
                },
        }
    Once the map is generated, for each case, we concatinate the header with the activity to name the columns and assign them the values
    For each case a new row in the dataframe is generated
    
    :param decision_point_activity: id of the activity inmediatly previous to the decision point which "why" wants to be discovered
    :type decision_point_activity: str
    :param target_label: name of the column where classification target label is stored
    :type target_label: str
    :param special_colnames: dict that contains the column names that refers to special columns: "Screenshot", "Variant", "Case"...
    :type special_colnames: dict
    :param columns_to_drop: Names of the colums to remove from the dataset
    :type columns_to_drop: list
    :param log_path: path where ui log to be processed is stored
    :type log_path: str
    :param path_dataset_saved: path where files that results from the flattening are stored
    :type path_dataset_saved: str
    :param actions_columns: list that contains column names that wont be added to the event information just before the decision point
    :type actions_columns: list
    """
    decision_point_activity = execution.extract_training_dataset.decision_point_activity
    target_label = execution.extract_training_dataset.target_label
    special_colnames = execution.case_study.special_colnames
    actions_columns = execution.extract_training_dataset.columns_to_drop_before_decision_point
    
    
    tprint("  " + PLATFORM_NAME + " - " + FLATTENING_PHASE_NAME, "fancy60")
    print(log_path+"\n")

    log = read_ui_log_as_dataframe(log_path)
    process_columns = [special_colnames["Case"], 
                       special_colnames["Activity"], 
                       special_colnames["Variant"],
                       special_colnames["Timestamp"], 
                       special_colnames["Screenshot"]]
    
    columns = list(log.columns)
    for c in process_columns:
        if c in columns:
            columns.remove(c)
        
    # Stablish common columns and the rest of the columns are concatinated with "_" + activity
    flat_dataset_row(log, columns, target_label, root_path+'_results', special_colnames["Case"], special_colnames["Activity"], 
                     special_colnames["Timestamp"], decision_point_activity, actions_columns)

                     
def decision_tree_training(log_path, path, execution):
    # "media/flattened_dataset.json",
    # "media", 
    # "sklearn",
    # ['ID3', 'CART', 'CHAID', 'C4.5'],
    # ["Timestamp_start", "Timestamp_end"],
    # 'Variant',
    # ['NameApp']
                           
    target_label = execution.extract_training_dataset.target_label
    flattened_json_log_path = os.path.join(path+"_results", 'flattened_dataset.json')
    implementation = execution.decision_tree_training.library
    configuration = execution.decision_tree_training.configuration
    columns_to_ignore = execution.decision_tree_training.columns_to_drop_before_decision_point
    one_hot_columns = execution.decision_tree_training.one_hot_columns
    k_fold_cross_validation = configuration["cv"] if "cv" in configuration else 3
    algorithms = configuration["algorithms"] if "algorithms" in configuration else None
    centroid_threshold = int(configuration["centroid_threshold"]) if "centroid_threshold" in configuration else None
    feature_values = configuration["feature_values"] if "feature_values" in configuration else None
    
    tprint(PLATFORM_NAME + " - " + DECISION_MODEL_DISCOVERY_PHASE_NAME, "fancy60")
    print(flattened_json_log_path+"\n")
    
    flattened_dataset = pd.read_json(flattened_json_log_path, orient ='index')
    # flattened_dataset.to_csv(path + "flattened_dataset.csv")    
    
    if not os.path.exists(os.path.join(path, DECISION_FOLDERNAME)):
        os.mkdir(os.path.join(path, DECISION_FOLDERNAME))
    
    # for col in flattened_dataset.columns:
    #     if "Coor" in col:
    #         columns_to_ignore.append(col)  
    
    # TODO: get type of TextInput column using NLP: convert to categorical variable (conversation, name, email, number, date, etc)
    flattened_dataset = flattened_dataset.drop(columns_to_ignore, axis=1)
    flattened_dataset.to_csv(os.path.join(path+"_results",FLATTENED_DATASET_NAME+".csv"))
    columns_len = flattened_dataset.shape[1]
    flattened_dataset = flattened_dataset.fillna('NaN')
    # tree_levels = {}
    
    if implementation == 'sklearn':
        res, times = sklearn_decision_tree(flattened_dataset, path+"_results", configuration, one_hot_columns, target_label, k_fold_cross_validation)
    elif implementation == 'chefboost':
        res, times = chefboost_decision_tree(flattened_dataset, path+"_results", algorithms, target_label, k_fold_cross_validation)
        # TODO: caculate number of tree levels automatically
        # for alg in algorithms:
            # rules_info = open(path+alg+'-rules.json')
            # rules_info_json = json.load(rules_info)
            # tree_levels[alg] = len(rules_info_json.keys())            
    else:
        raise Exception(_("Decision model chosen is not an option"))
    
    if feature_values:
        fe_checker = decision_tree_feature_checker(feature_values, centroid_threshold, path+"_results")
    else:
        fe_checker = None
    return res, fe_checker, times, columns_len#, tree_levels
    
def decision_tree_predict(module_path, instance):
    """
    moduleName = "outputs/rules/rules" #this will load outputs/rules/rules.py
    instance = for example ['Sunny', 'Hot', 'High', 'Weak']
    """
    tree = chef.restoreTree(module_path)
    prediction = tree.findDecision(instance)
    return prediction

class ExtractTrainingDatasetCreateView(CreateView):
    model = ExtractTrainingDataset
    form_class = ExtractTrainingDatasetForm
    template_name = "extract_training_dataset/create.html"
    
    def get_context_data(self, **kwargs):
        context = super(ExtractTrainingDatasetCreateView, self).get_context_data(**kwargs)
        context['case_study_id'] = self.kwargs.get('case_study_id')
        return context

    def form_valid(self, form):
        if not self.request.user.is_authenticated:
            raise ValidationError(_("User must be authenticated."))
        self.object = form.save(commit=False)
        self.object.user = self.request.user
        self.object.case_study = CaseStudy.objects.get(pk=self.kwargs.get('case_study_id'))
        saved = self.object.save()
        return HttpResponseRedirect(self.get_success_url())

class ExtractTrainingDatasetListView(ListView):
    model = ExtractTrainingDataset
    template_name = "extract_training_dataset/list.html"
    paginate_by = 50

    def get_context_data(self, **kwargs):
        context = super(ExtractTrainingDatasetListView, self).get_context_data(**kwargs)
        context['case_study_id'] = self.kwargs.get('case_study_id')
        return context

    def get_queryset(self):
        # Obtiene el ID del Experiment pasado como parámetro en la URL
        case_study_id = self.kwargs.get('case_study_id')

        # Filtra los objetos por case_study_id
        queryset = ExtractTrainingDataset.objects.filter(case_study__id=case_study_id, case_study__user=self.request.user).order_by('-created_at')

        return queryset
    

class ExtractTrainingDatasetDetailView(FormMixin, DetailView):
    model = ExtractTrainingDataset
    form_class = ExtractTrainingDatasetForm
    template_name = "extract_training_dataset/details.html"

    pk_url_kwarg = "extract_training_dataset_id"
    
    def get_context_data(self, **kwargs):
        context = super(ExtractTrainingDatasetDetailView, self).get_context_data(**kwargs)
        context['case_study_id'] = self.kwargs.get('case_study_id')
        context['form'] = ExtractTrainingDatasetForm(initial=model_to_dict(self.object))
        return context

    # def get_object(self, *args, **kwargs):
    #     extract_training_dataset = get_object_or_404(ExtractTrainingDataset, id=kwargs["extract_training_dataset_id"])
    #     return extract_training_dataset
        # return render(request, "extract_training_dataset/details.html", {"extract_training_dataset": extract_training_dataset, "case_study_id": kwargs["case_study_id"]})

def set_as_extracting_training_dataset_active(request):
    extracting_training_dataset_id = request.GET.get("extract_training_dataset_id")
    case_study_id = request.GET.get("case_study_id")
    extracting_training_dataset_list = ExtractTrainingDataset.objects.filter(case_study_id=case_study_id)
    for m in extracting_training_dataset_list:
        m.active = False
        m.save()
    extracting_training_dataset = ExtractTrainingDataset.objects.get(id=extracting_training_dataset_id)
    extracting_training_dataset.active = True
    extracting_training_dataset.save()
    return HttpResponseRedirect(reverse("decisiondiscovery:extract_training_dataset_list", args=[case_study_id]))

def set_as_extracting_training_dataset_inactive(request):
    extracting_training_dataset_id = request.GET.get("extract_training_dataset_id")
    case_study_id = request.GET.get("case_study_id")
    # Validations
    if not request.user.is_authenticated:
        raise ValidationError(_("User must be authenticated."))
    if CaseStudy.objects.get(pk=case_study_id).user != request.user:
        raise ValidationError(_("Case Study doesn't belong to the authenticated user."))
    if ExtractTrainingDataset.objects.get(pk=extracting_training_dataset_id).user != request.user:  
        raise ValidationError(_("Extracting_training_dataset doesn't belong to the authenticated user."))
    if ExtractTrainingDataset.objects.get(pk=extracting_training_dataset_id).case_study != CaseStudy.objects.get(pk=case_study_id):
        raise ValidationError(_("Extracting_training_dataset doesn't belong to the Case Study."))
    extracting_training_dataset = ExtractTrainingDataset.objects.get(id=extracting_training_dataset_id)
    extracting_training_dataset.active = False
    extracting_training_dataset.save()
    return HttpResponseRedirect(reverse("decisiondiscovery:extract_training_dataset_list", args=[case_study_id]))
    
def delete_extracting_training_dataset(request):
    extracting_training_dataset_id = request.GET.get("extract_training_dataset_id")
    case_study_id = request.GET.get("case_study_id")
    extracting_training_dataset = ExtractTrainingDataset.objects.get(id=extracting_training_dataset_id)
    if request.user.id != extracting_training_dataset.user.id:
        raise Exception("This object doesn't belong to the authenticated user")
    extracting_training_dataset.delete()
    return HttpResponseRedirect(reverse("decisiondiscovery:extract_training_dataset_list", args=[case_study_id]))

    
class DecisionTreeTrainingCreateView(CreateView):
    model = DecisionTreeTraining
    form_class = DecisionTreeTrainingForm
    template_name = "decision_tree_training/create.html"
    
    def get_context_data(self, **kwargs):
        context = super(DecisionTreeTrainingCreateView, self).get_context_data(**kwargs)
        context['case_study_id'] = self.kwargs.get('case_study_id')
        return context

    def form_valid(self, form):
        if not self.request.user.is_authenticated:
            raise ValidationError(_("User must be authenticated."))
        self.object = form.save(commit=False)
        self.object.user = self.request.user
        self.object.case_study = CaseStudy.objects.get(pk=self.kwargs.get('case_study_id'))
        saved = self.object.save()
        return HttpResponseRedirect(self.get_success_url())

class DecisionTreeTrainingListView(ListView):
    model = DecisionTreeTraining
    template_name = "decision_tree_training/list.html"
    paginate_by = 50

    def get_context_data(self, **kwargs):
        context = super(DecisionTreeTrainingListView, self).get_context_data(**kwargs)
        context['case_study_id'] = self.kwargs.get('case_study_id')
        return context

    def get_queryset(self):
        # Obtiene el ID del Experiment pasado como parámetro en la URL
        case_study_id = self.kwargs.get('case_study_id')

        # Filtra los objetos DecisionTreeTraining por case_study_id
        queryset = DecisionTreeTraining.objects.filter(case_study__id=case_study_id, case_study__user=self.request.user).order_by('-created_at')

        return queryset


class DecisionTreeTrainingDetailView(DetailView):
    def get(self, request, *args, **kwargs):
        decision_tree_training = get_object_or_404(DecisionTreeTraining, id=kwargs["decision_tree_training_id"])
        return render(request, "decision_tree_training/detail.html", {"decision_tree_training": decision_tree_training, "case_study_id": kwargs["case_study_id"]})

def set_as_decision_tree_training_active(request):
    decision_tree_training_id = request.GET.get("decision_tree_training_id")
    case_study_id = request.GET.get("case_study_id")
    decision_tree_training_list = DecisionTreeTraining.objects.filter(case_study_id=case_study_id)
    for m in decision_tree_training_list:
        m.active = False
        m.save()
    decision_tree_training = DecisionTreeTraining.objects.get(id=decision_tree_training_id)
    decision_tree_training.active = True
    decision_tree_training.save()
    return HttpResponseRedirect(reverse("decisiondiscovery:decision_tree_training_list", args=[case_study_id]))

def set_as_decision_tree_training_inactive(request):
    decision_tree_training_id = request.GET.get("decision_tree_training_id")
    case_study_id = request.GET.get("case_study_id")
    # Validations
    if not request.user.is_authenticated:
        raise ValidationError(_("User must be authenticated."))
    if CaseStudy.objects.get(pk=case_study_id).user != request.user:
        raise ValidationError(_("Case Study doesn't belong to the authenticated user."))
    if DecisionTreeTraining.objects.get(pk=decision_tree_training_id).user != request.user:  
        raise ValidationError(_("Decision Tree Training doesn't belong to the authenticated user."))
    if DecisionTreeTraining.objects.get(pk=decision_tree_training_id).case_study != CaseStudy.objects.get(pk=case_study_id):
        raise ValidationError(_("Decision Tree Training doesn't belong to the Case Study."))
    decision_tree_training = DecisionTreeTraining.objects.get(id=decision_tree_training_id)
    decision_tree_training.active = False
    decision_tree_training.save()
    return HttpResponseRedirect(reverse("decisiondiscovery:decision_tree_training_list", args=[case_study_id]))
    
def delete_decision_tree_training(request):
    decision_tree_training_id = request.GET.get("decision_tree_training_id")
    case_study_id = request.GET.get("case_study_id")
    decision_tree_training = DecisionTreeTraining.objects.get(id=decision_tree_training_id)
    if request.user.id != decision_tree_training.user.id:
        raise Exception(_("This object doesn't belong to the authenticated user"))
    decision_tree_training.delete()
    return HttpResponseRedirect(reverse("decisiondiscovery:decision_tree_training_list", args=[case_study_id]))

def decision_tree_feature_checker(feature_values, centroid_threshold, path):
    """
    
    A function to check conditions over decision tree representations

    Args:
        feature_values (dict): Classes and values of the features that should appear in the decision tree to reach this class
        
        Ejemplo:
         "feature_values": {
                    "1": {
                    "sta_enabled_717.5-606.5_2_B": 0.3,
                    "sta_checked_649.0-1110.5_4_D": 0.2
                    },
                    "2": {
                        "sta_enabled_717.5-606.5_2_B": 0.7,
                        "sta_checked_649.0-1110.5_4_D": 0.2
                    }
                }

    Returns:
        boolean: indicates if the path drives to the correct class

        dict: indicates how many times a feature appears in the decision tree. Example: {
            1: {
                'status_categorical__sta_enabled_717.5-606.5_2_B': 2,
                'status_categorical__sta_checked_649.0-1110.5_4_D': 1
            },
            2: {
                'status_categorical__sta_enabled_717.5-606.5_2_B': 1
            }
        }
    """
    dt_file = os.path.join(path, "decision_tree.log")
    
    metadata = {}        
    for target_class, fe_values_class in feature_values.items():
        tree, max_depth = parse_decision_tree(dt_file)
        path_exists, features_in_tree = find_path_in_decision_tree(tree, fe_values_class, target_class, centroid_threshold)
        metadata[target_class] = features_in_tree
        metadata[target_class]["tree_depth"] = max_depth
        metadata[target_class]["cumple_condicion"] = path_exists
    # print(path_exists)
    # print((len(features_in_tree) / len(feature_values))*100)
    return metadata