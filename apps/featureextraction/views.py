from apps.featureextraction.classification import legacy_ui_elements_classification, uied_ui_elements_classification
from art import tprint
from core.settings import platform_name, classification_phase_name, feature_extraction_phase_name
from apps.analyzer.utils import detect_fe_function
from django.http import HttpResponseRedirect
from django.views.generic import ListView, DetailView, CreateView
from django.core.exceptions import ValidationError
from .models import FeatureExtractionTechnique, UIElementsClassification, UIElementsDetection
from .forms import FeatureExtractionTechniqueForm, UIElementsClassificationForm, UIElementsDetectionForm

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
        case _:
            pass
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
        return FeatureExtractionTechnique.objects.all()
    
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
        return UIElementsClassification.objects.all()
    
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
        return UIElementsDetection.objects.all()
