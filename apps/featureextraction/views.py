from art import tprint
from core.settings import platform_name, classification_phase_name, feature_extraction_phase_name, aggregate_feature_extraction_phase_name
from apps.analyzer.utils import detect_fe_function, detect_agg_fe_function
from django.http import HttpResponseRedirect
from django.views.generic import ListView, DetailView, CreateView
from django.core.exceptions import ValidationError
from apps.analyzer.models import FeatureExtractionTechnique
from apps.analyzer.forms import FeatureExtractionTechniqueForm
from apps.featureextraction.SOM.classification import legacy_ui_elements_classification, uied_ui_elements_classification
from .models import UIElementsClassification, UIElementsDetection, Prefilters, Postfilters
from .forms import UIElementsClassificationForm, UIElementsDetectionForm, PrefiltersForm, PostfiltersForm

def ui_elements_classification(*data):
    # Classification can be done with different algorithms
    data_list = list(data)
    classifier_type = data_list.pop()
    data = tuple(data_list)

    tprint(platform_name + " - " + classification_phase_name, "fancy60")
    print(data_list[4]+"\n")
    
    match classifier_type:
        case "rpa-us":
            output = legacy_ui_elements_classification(*data)
        case "uied":
            output = uied_ui_elements_classification(*data)
        case "sam": #TODO
            output = legacy_ui_elements_classification(*data)
        case _:
            raise Exception("You select a type of UI element classification that doesnt exists")
    return output

def feature_extraction_technique(*data):
    tprint(platform_name + " - " + feature_extraction_phase_name, "fancy60")

    data_list = list(data)
    feature_extraction_technique_name = data_list.pop()
    skip = data_list.pop()
    data = tuple(data_list)
    output = None

    print("Feature extraction selected: " + feature_extraction_technique_name+"\n")
    
    if not skip:
        output = detect_fe_function(feature_extraction_technique_name)(*data)
    return output

def aggregate_features_as_dataset_columns(*data):
    tprint(platform_name + " - " + aggregate_feature_extraction_phase_name, "fancy60")

    data_list = list(data)
    agg_feature_extraction_technique_name = data_list.pop()
    skip = data_list.pop()
    data = tuple(data_list)
    output = None

    print("Aggregate feature extraction selected: " + agg_feature_extraction_technique_name+"\n")
    
    if not skip:
        output = detect_agg_fe_function(agg_feature_extraction_technique_name)(*data)
    return output

class FeatureExtractionTechniqueCreateView(CreateView):
    model = FeatureExtractionTechnique
    form_class = FeatureExtractionTechniqueForm
    template_name = "feature_extraction_technique/create.html"

    def form_valid(self, form):
        if not self.request.user.is_authenticated:
            raise ValidationError("User must be authenticated.")
        self.object = form.save(commit=False)
        self.object.user = self.request.user
        saved = self.object.save()
        return HttpResponseRedirect(self.get_success_url())

class FeatureExtractionTechniqueListView(ListView):
    model = FeatureExtractionTechnique
    template_name = "feature_extraction_technique/list.html"
    paginate_by = 50

    def get_queryset(self):
        return FeatureExtractionTechnique.objects.filter(user=self.request.user)
    
class UIElementsClassificationCreateView(CreateView):
    model = UIElementsClassification
    form_class = UIElementsClassificationForm
    template_name = "ui_elements_classification/create.html"

    def form_valid(self, form):
        if not self.request.user.is_authenticated:
            raise ValidationError("User must be authenticated.")
        self.object = form.save(commit=False)
        self.object.user = self.request.user
        saved = self.object.save()
        return HttpResponseRedirect(self.get_success_url())

class UIElementsClassificationListView(ListView):
    model = UIElementsClassification
    template_name = "ui_elements_classification/list.html"
    paginate_by = 50

    def get_queryset(self):
        return UIElementsClassification.objects.filter(user=self.request.user)
    
class UIElementsDetectionCreateView(CreateView):
    model = UIElementsDetection
    form_class = UIElementsDetectionForm
    template_name = "ui_elements_detection/create.html"

    def form_valid(self, form):
        if not self.request.user.is_authenticated:
            raise ValidationError("User must be authenticated.")
        self.object = form.save(commit=False)
        self.object.user = self.request.user
        saved = self.object.save()
        return HttpResponseRedirect(self.get_success_url())

class UIElementsDetectionListView(ListView):
    model = UIElementsDetection
    template_name = "ui_elements_detection/list.html"
    paginate_by = 50

    def get_queryset(self):
        return UIElementsDetection.objects.filter(user=self.request.user)


class PrefiltersCreateView(CreateView):
    model = Prefilters
    form_class = PrefiltersForm
    template_name = "prefiltering/create.html"

    def form_valid(self, form):
        if not self.request.user.is_authenticated:
            raise ValidationError("User must be authenticated.")
        self.object = form.save(commit=False)
        self.object.user = self.request.user
        saved = self.object.save()
        return HttpResponseRedirect(self.get_success_url())

class PrefiltersListView(ListView):
    model = Prefilters
    template_name = "prefiltering/list.html"
    paginate_by = 50

    def get_queryset(self):
        return Prefilters.objects.filter(user=self.request.user)

class PostfiltersCreateView(CreateView):
    model = Postfilters
    form_class = PostfiltersForm
    template_name = "postfiltering/create.html"

    def form_valid(self, form):
        if not self.request.user.is_authenticated:
            raise ValidationError("User must be authenticated.")
        self.object = form.save(commit=False)
        self.object.user = self.request.user
        saved = self.object.save()
        return HttpResponseRedirect(self.get_success_url())

class PostfiltersListView(ListView):
    model = Postfilters
    template_name = "postfiltering/list.html"
    paginate_by = 50

    def get_queryset(self):
        return Postfilters.objects.filter(user=self.request.user)
    