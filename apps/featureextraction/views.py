from art import tprint
from core.settings import sep, platform_name, classification_phase_name, feature_extraction_phase_name, aggregate_feature_extraction_phase_name
from django.http import HttpResponseRedirect, HttpResponse
from django.views.generic import ListView, DetailView, CreateView
from django.core.exceptions import ValidationError
from apps.analyzer.models import CaseStudy
from apps.featureextraction.SOM.classification import legacy_ui_elements_classification, uied_ui_elements_classification
from .models import UIElementsClassification, UIElementsDetection, Prefilters, Postfilters, FeatureExtractionTechnique
from .forms import UIElementsClassificationForm, UIElementsDetectionForm, PrefiltersForm, PostfiltersForm, FeatureExtractionTechniqueForm
from .relevantinfoselection.postfilters import draw_postfilter_relevant_ui_compos_borders
from .utils import detect_fe_function, detect_agg_fe_function
from .utils import draw_ui_compos_borders
from rest_framework import status


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
        case "fast-sam": #TODO
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

    def get_context_data(self, **kwargs):
        context = super(FeatureExtractionTechniqueListView, self).get_context_data(**kwargs)
        context['case_study_id'] = self.kwargs.get('case_study_id')
        return context

    def get_queryset(self):
        # Obtiene el ID del Experiment pasado como parámetro en la URL
        case_study_id = self.kwargs.get('case_study_id')

        # Filtra los objetos por case_study_id
        queryset = FeatureExtractionTechnique.objects.filter(case_study__id=case_study_id, case_study__user=self.request.user).order_by('-created_at')

        return queryset
    
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

    def get_context_data(self, **kwargs):
        context = super(UIElementsClassificationListView, self).get_context_data(**kwargs)
        context['case_study_id'] = self.kwargs.get('case_study_id')
        return context

    def get_queryset(self):
        # Obtiene el ID del Experiment pasado como parámetro en la URL
        case_study_id = self.kwargs.get('case_study_id')

        # Filtra los objetos por case_study_id
        queryset = UIElementsClassification.objects.filter(case_study__id=case_study_id, case_study__user=self.request.user).order_by('-created_at')

        return queryset
    
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

    def get_context_data(self, **kwargs):
        context = super(UIElementsDetectionListView, self).get_context_data(**kwargs)
        context['case_study_id'] = self.kwargs.get('case_study_id')
        return context

    def get_queryset(self):
        # Obtiene el ID del Experiment pasado como parámetro en la URL
        case_study_id = self.kwargs.get('case_study_id')

        # Filtra los objetos por case_study_id
        queryset = UIElementsDetection.objects.filter(case_study__id=case_study_id, case_study__user=self.request.user).order_by('-created_at')

        return queryset


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

    def get_context_data(self, **kwargs):
        context = super(PrefiltersListView, self).get_context_data(**kwargs)
        context['case_study_id'] = self.kwargs.get('case_study_id')
        return context

    def get_queryset(self):
        # Obtiene el ID del Experiment pasado como parámetro en la URL
        case_study_id = self.kwargs.get('case_study_id')

        # Filtra los objetos por case_study_id
        queryset = Prefilters.objects.filter(case_study__id=case_study_id, case_study__user=self.request.user).order_by('-created_at')

        return queryset

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

    def get_context_data(self, **kwargs):
        context = super(PostfiltersListView, self).get_context_data(**kwargs)
        context['case_study_id'] = self.kwargs.get('case_study_id')
        return context

    def get_queryset(self):
        # Obtiene el ID del Experiment pasado como parámetro en la URL
        case_study_id = self.kwargs.get('case_study_id')

        # Filtra los objetos por case_study_id
        queryset = Postfilters.objects.filter(case_study__id=case_study_id, case_study__user=self.request.user).order_by('-created_at')

        return queryset
    
    
def draw_postfilter(request, case_study_id):
    st = status.HTTP_200_OK
    
    try:
        case_study = CaseStudy.objects.get(id=case_study_id)
        if case_study.executed:
            # user = request.user
            # cs = CaseStudy.objects.filter(user=user, id=case_study_id)
            # if cs.exists() :
            # cs = cs[0]
            cs = case_study
            for scenario in cs.scenarios_to_study:
                draw_postfilter_relevant_ui_compos_borders(cs.exp_folder_complete_path + sep + scenario)

            # else:
            #     raise Exception("You don't have permissions to access this files")
            response = 'Postfiltered UI compos borders has been drawn!'
        else:
            response = 'The processing of this case study has not yet finished, please try again in a few minutes'

    except Exception as e:
        response = f"Case Study with id {case_study_id} raise an exception: " + str(e)
        st = status.HTTP_404_NOT_FOUND

    return HttpResponse(response, status=st)

def draw_ui_compos(request, case_study_id):
    st = status.HTTP_200_OK
    
    try:
        case_study = CaseStudy.objects.get(id=case_study_id)
        if case_study.executed:
            # user = request.user
            # cs = CaseStudy.objects.filter(user=user, id=case_study_id)
            # if cs.exists() :
            # cs = cs[0]
            cs = case_study
            for scenario in cs.scenarios_to_study:
                draw_ui_compos_borders(cs.exp_folder_complete_path + sep + scenario)

            # else:
            #     raise Exception("You don't have permissions to access this files")
            response = 'Postfiltered UI compos borders has been drawn!'
        else:
            response = 'The processing of this case study has not yet finished, please try again in a few minutes'

    except Exception as e:
        response = f"Case Study with id {case_study_id} raise an exception: " + str(e)
        st = status.HTTP_404_NOT_FOUND

    return HttpResponse(response, status=st)