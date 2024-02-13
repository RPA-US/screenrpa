from art import tprint
from core.settings import sep, PLATFORM_NAME, CLASSIFICATION_PHASE_NAME, FEATURE_EXTRACTION_PHASE_NAME, AGGREGATE_FEATURE_EXTRACTION_PHASE_NAME
from django.http import HttpResponseRedirect, HttpResponse
from django.views.generic import ListView, DetailView, CreateView
from apps.utils import MultiFormsView
from django.core.exceptions import ValidationError
from apps.analyzer.models import CaseStudy
from apps.featureextraction.SOM.classification import legacy_ui_elements_classification, uied_ui_elements_classification
from .models import UIElementsClassification, UIElementsDetection, Prefilters, Postfilters, FeatureExtractionTechnique
from .forms import UIElementsClassificationForm, UIElementsDetectionForm, PrefiltersForm, PostfiltersForm, FeatureExtractionTechniqueForm
from .relevantinfoselection.postfilters import draw_postfilter_relevant_ui_compos_borders
from .utils import detect_fe_function, detect_agg_fe_function
from .utils import draw_ui_compos_borders
from django.shortcuts import render, get_object_or_404
from django.urls import reverse
from rest_framework import status
from apps.analyzer.models import CaseStudy, Execution

def ui_elements_classification(*data):
    # Classification can be done with different algorithms
    data_list = list(data)
    classifier_type = data_list.pop()
    data = tuple(data_list)

    tprint(PLATFORM_NAME + " - " + CLASSIFICATION_PHASE_NAME, "fancy60")
    print(data_list[4]+"\n")
    
    match classifier_type:
        case "rpa-us":
            output = legacy_ui_elements_classification(*data)
        case "uied":
            output = uied_ui_elements_classification(*data)
        case "sam":
            output = legacy_ui_elements_classification(*data)
        case "fast-sam":
            output = legacy_ui_elements_classification(*data)
        case "screen2som":
            output = None
        case _:
            raise Exception("You select a type of UI element classification that doesnt exists")
    return output

def feature_extraction_technique(*data):
    tprint(PLATFORM_NAME + " - " + FEATURE_EXTRACTION_PHASE_NAME, "fancy60")

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
    tprint(PLATFORM_NAME + " - " + AGGREGATE_FEATURE_EXTRACTION_PHASE_NAME, "fancy60")

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
    
    def get_context_data(self, **kwargs):
        context = super(FeatureExtractionTechniqueCreateView, self).get_context_data(**kwargs)
        context['case_study_id'] = self.kwargs.get('case_study_id')
        return context

    def form_valid(self, form):
        if not self.request.user.is_authenticated:
            raise ValidationError("User must be authenticated.")
        self.object = form.save(commit=False)
        self.object.user = self.request.user
        self.object.case_study = CaseStudy.objects.get(pk=self.kwargs.get('case_study_id'))
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

class FeatureExtractionTechniqueDetailView(DetailView):
    def get(self, request, *args, **kwargs):
        feature_extraction_technique = get_object_or_404(FeatureExtractionTechnique, id=kwargs["feature_extraction_technique_id"])
        return render(request, "feature_extraction_technique/detail.html", {"feature_extraction_technique": feature_extraction_technique, "case_study_id": kwargs["case_study_id"]})

def set_as_feature_extraction_technique_active(request):
    feature_extraction_technique_id = request.GET.get("feature_extraction_technique_id")
    case_study_id = request.GET.get("case_study_id")
    feature_extraction_technique_list = FeatureExtractionTechnique.objects.filter(case_study_id=case_study_id)
    for m in feature_extraction_technique_list:
        m.active = False
        m.save()
    feature_extraction_technique = FeatureExtractionTechnique.objects.get(id=feature_extraction_technique_id)
    feature_extraction_technique.active = True
    feature_extraction_technique.save()
    return HttpResponseRedirect(reverse("featureextraction:fe_technique_list", args=[case_study_id]))
    
def delete_feature_extraction_technique(request):
    feature_extraction_technique_id = request.GET.get("feature_extraction_technique_id")
    case_study_id = request.GET.get("case_study_id")
    feature_extraction_technique = FeatureExtractionTechnique.objects.get(id=feature_extraction_technique_id)
    if request.user.id != feature_extraction_technique.user.id:
        raise Exception("This object doesn't belong to the authenticated user")
    feature_extraction_technique.delete()
    return HttpResponseRedirect(reverse("featureextraction:fe_technique_list", args=[case_study_id]))


class UIElementsDetectionCreateView(MultiFormsView):
    form_classes = {
        'ui_elements_detection': UIElementsDetectionForm,
        'ui_elements_classification': UIElementsClassificationForm,
    }
    template_name = "ui_elements_detection/create.html"
    # Current url is /new/<id>/ so we need to redirect to /list/<id>

    def get_context_data(self, **kwargs):
        context = super(UIElementsDetectionCreateView, self).get_context_data(**kwargs)
        context['case_study_id'] = self.kwargs.get('case_study_id')
        return context

    def forms_valid(self, forms):
        ui_elements_detection_form = forms['ui_elements_detection']
        ui_elements_classification_form = forms['ui_elements_classification']
        ui_elem_det_obj = self.ui_elements_detection_form_valid(ui_elements_detection_form)
        self.ui_elements_classification_form_valid(ui_elements_classification_form, ui_elem_det_obj)
        self.success_url = f"../../list/{self.kwargs.get('case_study_id')}"
        return HttpResponseRedirect(self.get_success_url())

    def ui_elements_detection_form_valid(self, form):
        if not self.request.user.is_authenticated:
            raise ValidationError("User must be authenticated.")
        self.object = form.save(commit=False)
        self.object.user = self.request.user
        self.object.case_study = CaseStudy.objects.get(pk=self.kwargs.get('case_study_id'))
        self.object.save()
        return self.object
    
    def ui_elements_classification_form_valid(self, form, ui_elem_det_obj):
        if not self.request.user.is_authenticated:
            raise ValidationError("User must be authenticated.")
        self.object = form.save(commit=False)
        if form.cleaned_data['model']:
            self.object.type = ui_elem_det_obj.type
            self.object.model = form.cleaned_data['model']
        else:
            return
        self.object.user = self.request.user
        self.object.case_study = CaseStudy.objects.get(pk=self.kwargs.get('case_study_id'))
        self.object.save()
        ui_elem_det_obj.ui_elements_classification = self.object
        ui_elem_det_obj.save()

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

class UIElementsDetectionDetailView(MultiFormsView):
    form_classes = {
        'ui_elements_detection': UIElementsDetectionForm,
        'ui_elements_classification': UIElementsClassificationForm,
    }
    template_name = "ui_elements_detection/details.html"
    # Current url is /new/<id>/ so we need to redirect to /list/<id>

    def get_context_data(self, **kwargs):
        context = super(UIElementsDetectionDetailView, self).get_context_data(**kwargs)
        context['case_study_id'] = self.kwargs.get('case_study_id')
        return context

    def get_ui_elements_detection_initial(self):
        ui_elements_detection = get_object_or_404(UIElementsDetection, id=self.kwargs["ui_elements_detection_id"])
        # Return serialized
        return {
            "initial": {
                "title": ui_elements_detection.title,
                "type": ui_elements_detection.type,
                "configurations": ui_elements_detection.configurations,
                "ocr": ui_elements_detection.ocr,
            },
            "instance": ui_elements_detection,
        }

    def get_ui_elements_classification_initial(self):
        ui_elements_detection = get_object_or_404(UIElementsDetection, id=self.kwargs["ui_elements_detection_id"])
        if ui_elements_detection.ui_elements_classification:
            ui_elements_classification = ui_elements_detection.ui_elements_classification
        else:
            ui_elements_classification = UIElementsClassification()
        # Return serialized
        return {
            "initial": {
                "model": ui_elements_classification.model,
            },
            "instance": ui_elements_classification,
        }
    
    def forms_valid(self, forms):
        ui_elements_detection_form = forms['ui_elements_detection']
        ui_elements_classification_form = forms['ui_elements_classification']
        ui_elem_det_obj = self.ui_elements_detection_form_valid(ui_elements_detection_form)
        self.ui_elements_classification_form_valid(ui_elements_classification_form, ui_elem_det_obj)
        self.success_url = f"../../../list/{self.kwargs.get('case_study_id')}"
        if ui_elem_det_obj.active:
            # Redirect to set as active with GET parameters case_study_id and ui_elem_detection_id
            self.success_url = f"../../../active/?case_study_id={self.kwargs.get('case_study_id')}&ui_elem_detection_id={ui_elem_det_obj.id}"
        return HttpResponseRedirect(self.get_success_url())

    def ui_elements_detection_form_valid(self, form):
        if not self.request.user.is_authenticated:
            raise ValidationError("User must be authenticated.")
        self.object = UIElementsDetection.objects.filter(pk=self.kwargs["ui_elements_detection_id"], user=self.request.user)
        if self.object.exists():
            self.object.update(**form.cleaned_data)
        else:
            raise ValidationError("This object doesn't belong to the authenticated user")
        return self.object.first()
    
    def ui_elements_classification_form_valid(self, form, ui_elem_det_obj):
        if not self.request.user.is_authenticated:
            raise ValidationError("User must be authenticated.")

        self.object = form.save(commit=False)
        if not ui_elem_det_obj.ui_elements_classification:
            if self.object.model == 'IGNORE':
                return

            self.object.user = self.request.user
            self.object.case_study = CaseStudy.objects.get(pk=self.kwargs.get('case_study_id'))
            self.object.save()
            ui_elem_det_obj.ui_elements_classification = self.object
            ui_elem_det_obj.save()
        elif self.object.model == 'IGNORE':
            UIElementsClassification.objects.filter(pk=ui_elem_det_obj.ui_elements_classification.id).delete()
        else:
            UIElementsClassification.objects.filter(pk=ui_elem_det_obj.ui_elements_classification.id).update(**form.cleaned_data)


def set_as_ui_elements_detection_active(request):
    ui_elements_detection_id = request.GET.get("ui_elem_detection_id")
    case_study_id = request.GET.get("case_study_id")
    prefilter_list = UIElementsDetection.objects.filter(case_study_id=case_study_id)
    for m in prefilter_list:
        m.active = False
        m.save()
    ui_elements_detection = UIElementsDetection.objects.get(id=ui_elements_detection_id)
    ui_elements_detection.active = True
    ui_elements_detection.save()
    
    # UI Elements Classification
    UIElementsClassification.objects.filter(case_study_id=case_study_id, active=True).update(active=False)
    if ui_elements_detection.ui_elements_classification:
        ui_elements_classification = ui_elements_detection.ui_elements_classification
        ui_elements_classification.active = True
        ui_elements_classification.save()

    return HttpResponseRedirect(reverse("featureextraction:ui_detection_list", args=[case_study_id]))
    
def delete_ui_elements_detection(request):
    ui_element_detection_id = request.GET.get("ui_elem_detection_id")
    case_study_id = request.GET.get("case_study_id")
    ui_elements_detection = UIElementsDetection.objects.get(id=ui_element_detection_id)
    if request.user.id != ui_elements_detection.user.id:
        raise Exception("This object doesn't belong to the authenticated user")
    ui_elements_detection.delete()
    return HttpResponseRedirect(reverse("featureextraction:ui_detection_list", args=[case_study_id]))


class PrefiltersCreateView(CreateView):
    model = Prefilters
    form_class = PrefiltersForm
    template_name = "prefiltering/create.html"
    
    def get_context_data(self, **kwargs):
        context = super(PrefiltersCreateView, self).get_context_data(**kwargs)
        context['case_study_id'] = self.kwargs.get('case_study_id')
        return context    

    def form_valid(self, form):
        if not self.request.user.is_authenticated:
            raise ValidationError("User must be authenticated.")
        self.object = form.save(commit=False)
        self.object.user = self.request.user
        self.object.case_study = CaseStudy.objects.get(pk=self.kwargs.get('case_study_id'))
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

    
class PrefiltersDetailView(DetailView):
    def get(self, request, *args, **kwargs):
        prefilter = get_object_or_404(Prefilters, id=kwargs["prefilter_id"])
        return render(request, "prefiltering/detail.html", {"prefilter": prefilter, "case_study_id": kwargs["case_study_id"]})

def set_as_prefilters_active(request):
    prefilter_id = request.GET.get("prefilter_id")
    case_study_id = request.GET.get("case_study_id")
    prefilter_list = Prefilters.objects.filter(case_study_id=case_study_id)
    for m in prefilter_list:
        m.active = False
        m.save()
    prefilter = Prefilters.objects.get(id=prefilter_id)
    prefilter.active = True
    prefilter.save()
    return HttpResponseRedirect(reverse("featureextraction:prefilters_list", args=[case_study_id]))
    
def delete_prefilter(request):
    prefilter_id = request.GET.get("prefilter_id")
    case_study_id = request.GET.get("case_study_id")
    prefilter = Prefilters.objects.get(id=prefilter_id)
    if request.user.id != prefilter.user.id:
        raise Exception("This object doesn't belong to the authenticated user")
    prefilter.delete()
    return HttpResponseRedirect(reverse("featureextraction:prefilters_list", args=[case_study_id]))



class PostfiltersCreateView(CreateView):
    model = Postfilters
    form_class = PostfiltersForm
    template_name = "postfiltering/create.html"
    
    def get_context_data(self, **kwargs):
        context = super(PostfiltersCreateView, self).get_context_data(**kwargs)
        context['case_study_id'] = self.kwargs.get('case_study_id')
        return context   

    def form_valid(self, form):
        if not self.request.user.is_authenticated:
            raise ValidationError("User must be authenticated.")
        self.object = form.save(commit=False)
        self.object.user = self.request.user
        self.object.case_study = CaseStudy.objects.get(pk=self.kwargs.get('case_study_id'))
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
    
class PostfiltersDetailView(DetailView):
    def get(self, request, *args, **kwargs):
        postfilter = get_object_or_404(Postfilters, id=kwargs["postfilter_id"])
        return render(request, "postfiltering/detail.html", {"postfilter": postfilter, "case_study_id": kwargs["case_study_id"]})

def set_as_postfilters_active(request):
    postfilter_id = request.GET.get("postfilter_id")
    case_study_id = request.GET.get("case_study_id")
    postfilter_list = Postfilters.objects.filter(case_study_id=case_study_id)
    for m in postfilter_list:
        m.active = False
        m.save()
    postfilter = Postfilters.objects.get(id=postfilter_id)
    postfilter.active = True
    postfilter.save()
    return HttpResponseRedirect(reverse("featureextraction:postfilters_list", args=[case_study_id]))
    
def delete_postfilter(request):
    postfilter_id = request.GET.get("postfilter_id")
    case_study_id = request.GET.get("case_study_id")
    postfilter = Postfilters.objects.get(id=postfilter_id)
    if request.user.id != postfilter.user.id:
        raise Exception("This object doesn't belong to the authenticated user")
    postfilter.delete()
    return HttpResponseRedirect(reverse("featureextraction:postfilters_list", args=[case_study_id]))

#############################################################################################################
###       DRAWING              ##############################################################################
#############################################################################################################

def draw_postfilter(request, execution_id):
    st = status.HTTP_200_OK
    
    try:
        execution = Execution.objects.get(id=execution_id)
        if execution.executed:
            # user = request.user
            # cs = CaseStudy.objects.filter(user=user, id=execution_id)
            # if cs.exists() :
            # cs = cs[0]
            for scenario in execution.scenarios_to_study:
                draw_postfilter_relevant_ui_compos_borders(execution.exp_folder_complete_path + sep + scenario)

            # else:
            #     raise Exception("You don't have permissions to access this files")
            response = 'Postfiltered UI compos borders has been drawn!'
        else:
            response = 'The processing of this case study has not yet finished, please try again in a few minutes'

    except Exception as e:
        response = f"Case Study with id {execution_id} raise an exception: " + str(e)
        st = status.HTTP_404_NOT_FOUND

    return HttpResponse(response, status=st)

def draw_ui_compos(request, execution_id):
    st = status.HTTP_200_OK
    
    try:
        execution = Execution.objects.get(id=execution_id)
        if execution.executed:
            # user = request.user
            # cs = CaseStudy.objects.filter(user=user, id=execution_id)
            # if cs.exists() :
            # cs = cs[0]
            for scenario in execution.scenarios_to_study:
                draw_ui_compos_borders(execution.exp_folder_complete_path + sep + scenario)

            # else:
            #     raise Exception("You don't have permissions to access this files")
            response = 'Postfiltered UI compos borders has been drawn!'
        else:
            response = 'The processing of this case study has not yet finished, please try again in a few minutes'

    except Exception as e:
        response = f"Case Study with id {execution_id} raise an exception: " + str(e)
        st = status.HTTP_404_NOT_FOUND

    return HttpResponse(response, status=st)