import csv
import json
import os
import random
from art import tprint
from core.settings import sep, PLATFORM_NAME, CLASSIFICATION_PHASE_NAME, SINGLE_FEATURE_EXTRACTION_PHASE_NAME, AGGREGATE_FEATURE_EXTRACTION_PHASE_NAME
from django.contrib.auth.mixins import LoginRequiredMixin
from django.contrib.auth.decorators import login_required
from django.http import HttpResponseRedirect, HttpResponse
from django.views.generic import ListView, DetailView, CreateView
from apps.utils import MultiFormsView
from django.core.exceptions import ValidationError
from apps.analyzer.models import CaseStudy
from apps.featureextraction.SOM.classification import legacy_ui_elements_classification, uied_ui_elements_classification
from .models import UIElementsClassification, UIElementsDetection, Prefilters, Postfilters, FeatureExtractionTechnique
from .forms import UIElementsClassificationForm, UIElementsDetectionForm, PrefiltersForm, PostfiltersForm, FeatureExtractionTechniqueForm
from .relevantinfoselection.postfilters import draw_postfilter_relevant_ui_compos_borders
from .utils import detect_single_fe_function, detect_agg_fe_function
from .utils import draw_ui_compos_borders
from django.shortcuts import render, get_object_or_404
from django.urls import reverse
from rest_framework import status
from apps.analyzer.models import CaseStudy, Execution
from django.utils.translation import gettext_lazy as _

def ui_elements_classification(log_path, path_scenario, execution):
    # Classification can be done with different algorithms
    tprint(PLATFORM_NAME + " - " + CLASSIFICATION_PHASE_NAME, "fancy60")
    print(path_scenario+"\n")

    if not execution.ui_elements_detection.preloaded:
        match execution.ui_elements_classification.type:
            case "rpa-us":
                output = legacy_ui_elements_classification(log_path, path_scenario, execution)
            case "uied":
                output = uied_ui_elements_classification(log_path, path_scenario, execution)
            case "sam":
                output = legacy_ui_elements_classification(log_path, path_scenario, execution)
            case "fast-sam":
                output = legacy_ui_elements_classification(log_path, path_scenario, execution)
            case "screen2som":
                output = None
            case _:
                raise Exception("You select a type of UI element classification that doesnt exists")
    else:
        output = None
    return output

def feature_extraction_technique(log_path, path_scenario, execution, fe):

    fe_type = fe.type
    feature_extraction_technique_name = fe.technique_name
    skip = fe.preloaded
    output = None
    
    if not skip:
        if fe_type == "SINGLE":
            tprint(PLATFORM_NAME + " - " + SINGLE_FEATURE_EXTRACTION_PHASE_NAME, "fancy60")
            print("Single feature extraction selected: " + feature_extraction_technique_name+"\n")
            output = detect_single_fe_function(feature_extraction_technique_name)(log_path, path_scenario, execution, fe)
        else:
            tprint(PLATFORM_NAME + " - " + AGGREGATE_FEATURE_EXTRACTION_PHASE_NAME, "fancy60")
            print("Aggregate feature extraction selected: " + feature_extraction_technique_name+"\n")
            output = detect_agg_fe_function(feature_extraction_technique_name)(log_path, path_scenario, execution, fe)
    return output

class FeatureExtractionTechniqueCreateView(CreateView, LoginRequiredMixin):
    login_url = "/login/"
    model = FeatureExtractionTechnique
    form_class = FeatureExtractionTechniqueForm
    template_name = "feature_extraction_technique/create.html"

    # Check if the the phase can be interacted with (included in case study available phases)
    def get(self, request, *args, **kwargs):
        if not self.request.user.is_authenticated:
            raise ValidationError(_("User must be authenticated."))
        case_study = CaseStudy.objects.get(pk=kwargs["case_study_id"])
        if 'FeatureExtractionTechnique' in case_study.available_phases:
            return super().get(request, *args, **kwargs)
        else:
            return HttpResponseRedirect(reverse("analyzer:casestudy_list"))
    
    def post(self, request, *args, **kwargs):
        if not self.request.user.is_authenticated:
            raise ValidationError(_("User must be authenticated."))
        case_study = CaseStudy.objects.get(pk=kwargs["case_study_id"])
        if 'FeatureExtractionTechnique' in case_study.available_phases:
            return super().post(request, *args, **kwargs)
        else:
            return HttpResponseRedirect(reverse("analyzer:casestudy_list"))
    
    def get_context_data(self, **kwargs):
        context = super(FeatureExtractionTechniqueCreateView, self).get_context_data(**kwargs)
        context['case_study_id'] = self.kwargs.get('case_study_id')

        # Load single and aggregate techniques from configurations
        single_json = json.load(open("configuration/single_feature_extractors.json"))
        aggregate_json = json.load(open("configuration/aggregate_feature_extractors.json"))
        context["options"] = {
            "single": single_json.items(),
            "aggregate": aggregate_json.items()
        }
        return context

    def form_valid(self, form):
        if not self.request.user.is_authenticated:
            raise ValidationError("User must be authenticated.")
        self.object = form.save(commit=False)
        self.object.user = self.request.user
        self.object.case_study = CaseStudy.objects.get(pk=self.kwargs.get('case_study_id'))
        saved = self.object.save()
        return HttpResponseRedirect(self.get_success_url())

class FeatureExtractionTechniqueListView(ListView, LoginRequiredMixin):
    login_url = "/login/"
    model = FeatureExtractionTechnique
    template_name = "feature_extraction_technique/list.html"
    paginate_by = 50

    # Check if the the phase can be interacted with (included in case study available phases)
    def get(self, request, *args, **kwargs):
        case_study = CaseStudy.objects.get(pk=kwargs["case_study_id"])
        if not case_study:
            return HttpResponse(status=404, content="Case Study not found.")
        elif case_study.user != request.user:
            return HttpResponse(status=403, content="Case Study doesn't belong to the authenticated user.")
        if 'FeatureExtractionTechnique' in case_study.available_phases:
            return super().get(request, *args, **kwargs)
        else:
            return HttpResponseRedirect(reverse("analyzer:casestudy_list"))

    def get_context_data(self, **kwargs):
        context = super(FeatureExtractionTechniqueListView, self).get_context_data(**kwargs)
        context['case_study_id'] = self.kwargs.get('case_study_id')
        return context

    def get_queryset(self):
        # Obtiene el ID del Experiment pasado como parámetro en la URL
        case_study_id = self.kwargs.get('case_study_id')

        # Search if s is a query parameter
        search = self.request.GET.get("s")
        # Filtra los objetos por case_study_id
        if search:
            queryset = FeatureExtractionTechnique.objects.filter(case_study__id=case_study_id, case_study__user=self.request.user, title__icontains=search).order_by('-created_at')
        else:
            queryset = FeatureExtractionTechnique.objects.filter(case_study__id=case_study_id, case_study__user=self.request.user).order_by('-created_at')

        # Filters by execution_id
        execution_id = self.request.GET.get("exec_id")
        if execution_id:
            queryset = queryset.filter(executions__id=execution_id)

        return queryset

class FeatureExtractionTechniqueDetailView(DetailView, LoginRequiredMixin):
    login_url = "/login/"
    # Check if the the phase can be interacted with (included in case study available phases)
 
    def get(self, request, *args, **kwargs):
        feature_extraction = get_object_or_404(FeatureExtractionTechnique, id=kwargs["feature_extraction_technique_id"])
        if not feature_extraction:
            return HttpResponse(status=404, content="FE not found.")
        elif feature_extraction.case_study.user != request.user:
            return HttpResponse(status=403, content="FE doesn't belong to the authenticated user.")

        form = FeatureExtractionTechniqueForm(read_only=True, instance=feature_extraction)
        if 'case_study_id' in kwargs:
            case_study = get_object_or_404(CaseStudy, id=kwargs['case_study_id'])
            if 'FeatureExtractionTechnique' in case_study.available_phases:
                context= {"feature_extraction_technique": feature_extraction, 
                    "case_study_id": case_study.id,
                    "form": form,}
        
                return render(request, "feature_extraction_technique/detail.html", context)
            else:
                return HttpResponseRedirect(reverse("analyzer:casestudy_list"))
         
        elif 'execution_id' in kwargs:
            execution = get_object_or_404(Execution, id=kwargs['execution_id'])
            if execution.feature_extraction_technique:
                context= {"feature_extraction_technique": feature_extraction, 
                            "execution_id": execution.id,
                            "form": form,}
            
                return render(request, "feature_extraction_technique/detail.html", context)
            else:
                return HttpResponseRedirect(reverse("analyzer:execution_list"))

@login_required(login_url="/login/")
def set_as_feature_extraction_technique_active(request):
    feature_extraction_technique_id = request.GET.get("feature_extraction_technique_id")
    case_study_id = request.GET.get("case_study_id")

    # Now we allow for more than one fe to be active
        # feature_extraction_technique_list = FeatureExtractionTechnique.objects.filter(case_study_id=case_study_id)
        # for m in feature_extraction_technique_list:
        #     m.active = False
        #     m.save()

    feature_extraction_technique = FeatureExtractionTechnique.objects.get(id=feature_extraction_technique_id)
    feature_extraction_technique.active = True
    feature_extraction_technique.save()
    return HttpResponseRedirect(reverse("featureextraction:fe_technique_list", args=[case_study_id]))

@login_required(login_url="/login/")
def set_as_feature_extraction_technique_inactive(request):
    feature_extraction_technique_id = request.GET.get("feature_extraction_technique_id")
    case_study_id = request.GET.get("case_study_id")
    # Validations
    if not request.user.is_authenticated:
        raise ValidationError(_("User must be authenticated."))
    if CaseStudy.objects.get(pk=case_study_id).user != request.user:
        raise ValidationError(_("Case Study doesn't belong to the authenticated user."))
    if FeatureExtractionTechnique.objects.get(pk=feature_extraction_technique_id).user != request.user:  
        raise ValidationError(_("Feature Extraction Technique doesn't belong to the authenticated user."))
    if FeatureExtractionTechnique.objects.get(pk=feature_extraction_technique_id).case_study != CaseStudy.objects.get(pk=case_study_id):
        raise ValidationError(_("Feature Extraction Technique Tree Training doesn't belong to the Case Study."))
    feature_extraction_technique = FeatureExtractionTechnique.objects.get(id=feature_extraction_technique_id)
    feature_extraction_technique.active = False
    feature_extraction_technique.save()
    return HttpResponseRedirect(reverse("featureextraction:fe_technique_list", args=[case_study_id]))
    
@login_required(login_url="/login/")
def delete_feature_extraction_technique(request):
    feature_extraction_technique_id = request.GET.get("feature_extraction_technique_id")
    case_study_id = request.GET.get("case_study_id")
    feature_extraction_technique = FeatureExtractionTechnique.objects.get(id=feature_extraction_technique_id)
    if request.user.id != feature_extraction_technique.user.id:
        raise Exception("This object doesn't belong to the authenticated user")
    feature_extraction_technique.delete()
    return HttpResponseRedirect(reverse("featureextraction:fe_technique_list", args=[case_study_id]))

#########################################################

class FeatureExtractionResultDetailView(DetailView, LoginRequiredMixin):
    login_url = "/login/"

    def get(self, request, *args, **kwargs):
        # Get the Execution object or raise a 404 error if not found
        execution = get_object_or_404(Execution, id=kwargs["execution_id"])     
        if not execution:
            return HttpResponse(status=404, content="FE not found.")
        elif execution.user != request.user:
            return HttpResponse(status=403, content="FE doesn't belong to the authenticated user.")

        scenario = request.GET.get('scenario')
        download = request.GET.get('download')

        if scenario == None:
            #scenario = "1"
            scenario = execution.scenarios_to_study[0] # by default, the first one that was indicated
      
        # TODO: Sujeto a cambios en la estructura de la carpeta
        #path_to_csv_file = execution.exp_folder_complete_path + "/"+ scenario +"/flattened_dataset.csv"
        path_to_csv_file = os.path.join(execution.exp_folder_complete_path, scenario+"_results", "flattened_dataset.csv")
        # CSV Download
        if path_to_csv_file and download=="True":
            return ResultDownload(path_to_csv_file)  
     
        # CSV Reading and Conversion to JSON
        csv_data_json = read_csv_to_json(path_to_csv_file)

        # Include CSV data in the context for the template
        context = {
            "execution_id": execution.id,
            "csv_data": csv_data_json,  # Data to be used in the HTML template
            "scenarios": execution.scenarios_to_study,
            "scenario": scenario
            }

        # Render the HTML template with the context including the CSV data
        return render(request, "feature_extraction_technique/result.html", context)


#############################################33
def read_csv_to_json(path_to_csv_file):
    # Initialize a list to hold the CSV data converted into dictionaries
    csv_data = []       
    # Check if the path to the CSV file exists and read the data
    try:
        with open(path_to_csv_file, 'r', newline='') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                csv_data.append(row)
    except FileNotFoundError:
        print(f"File not found: {path_to_csv_file}")
    # Convert csv_data to JSON
    csv_data_json = json.dumps(csv_data)
    return csv_data_json
##########################################3
def ResultDownload(path_to_csv_file):
    with open(path_to_csv_file, 'r', newline='') as csvfile:
        # Create an HTTP response with the content of the CSV
        response = HttpResponse(content_type='text/csv')
        response['Content-Disposition'] = 'inline; filename="{}"'.format(os.path.basename(path_to_csv_file))
        writer = csv.writer(response)
        reader = csv.reader(csvfile)
        for row in reader:
            writer.writerow(row)
        return response
    
#############################################################





class UIElementsDetectionCreateView(MultiFormsView, LoginRequiredMixin):
    login_url = "/login/"
    form_classes = {
        'ui_elements_detection': UIElementsDetectionForm,
        'ui_elements_classification': UIElementsClassificationForm,
    }
    template_name = "ui_elements_detection/create.html"
    # Current url is /new/<id>/ so we need to redirect to /list/<id>

    # Check if the the phase can be interacted with (included in case study available phases)
    def get(self, request, *args, **kwargs):
        case_study = CaseStudy.objects.get(pk=kwargs["case_study_id"])
        if not case_study:
            return HttpResponse(status=404, content="UI Elm Det not found.")
        elif case_study.user != request.user:
            return HttpResponse(status=403, content="UI Elm Det doesn't belong to the authenticated user.")

        if 'UIElementsDetection' in case_study.available_phases:
            return super().get(request, *args, **kwargs)
        else:
            return HttpResponseRedirect(reverse("analyzer:casestudy_list"))
    
    def post(self, request, *args, **kwargs):
        case_study = CaseStudy.objects.get(pk=kwargs["case_study_id"])
        if not case_study:
            return HttpResponse(status=404, content="UI Elm Det not found.")
        elif case_study.user != request.user:
            return HttpResponse(status=403, content="UI Elm Det doesn't belong to the authenticated user.")

        if 'UIElementsDetection' in case_study.available_phases:
            return super().post(request, *args, **kwargs)
        else:
            return HttpResponseRedirect(reverse("analyzer:casestudy_list"))
    

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

class UIElementsDetectionListView(LoginRequiredMixin, ListView):
    login_url = "/login/"
    model = UIElementsDetection
    template_name = "ui_elements_detection/list.html"
    paginate_by = 50

    # Check if the the phase can be interacted with (included in case study available phases)
    def get(self, request, *args, **kwargs):
        case_study = CaseStudy.objects.get(pk=kwargs["case_study_id"])
        if not case_study:
            return HttpResponse(status=404, content="UI Elm Det not found.")
        elif case_study.user != request.user:
            return HttpResponse(status=403, content="UI Elm Det doesn't belong to the authenticated user.")

        if 'UIElementsDetection' in case_study.available_phases:
            return super().get(request, *args, **kwargs)
        else:
            return self.super().get(request, *args, **kwargs)

    def get_context_data(self, **kwargs):
        context = super(UIElementsDetectionListView, self).get_context_data(**kwargs)
        context['case_study_id'] = self.kwargs.get('case_study_id')
        return context

    def get_queryset(self):
        # Obtiene el ID del Experiment pasado como parámetro en la URL
        case_study_id = self.kwargs.get('case_study_id')

        # Search if s is a query parameter
        search = self.request.GET.get("s")
        # Filtra los objetos por case_study_id
        if search:
            queryset = UIElementsDetection.objects.filter(case_study__id=case_study_id, case_study__user=self.request.user, title__icontains=search).order_by('-created_at')
        else:
            queryset = UIElementsDetection.objects.filter(case_study__id=case_study_id, case_study__user=self.request.user).order_by('-created_at')

        return queryset

class UIElementsDetectionDetailView(MultiFormsView, LoginRequiredMixin):
    login_url = "/login/"
    form_classes = {
        'ui_elements_detection': UIElementsDetectionForm,
        'ui_elements_classification': UIElementsClassificationForm,
    }
    template_name = "ui_elements_detection/details.html"
    # Current url is /new/<id>/ so we need to redirect to /list/<id>

    # Check if the the phase can be interacted with (included in case study available phases)
    def get(self, request, *args, **kwargs):
        if 'case_study_id' in self.kwargs:
            #context['case_study_id'] = self.kwargs['case_study_id']
            case_study = CaseStudy.objects.get(pk=kwargs["case_study_id"])
            if not case_study:
                return HttpResponse(status=404, content="UI Elm Det not found.")
            elif case_study.user != request.user:
                return HttpResponse(status=403, content="UI Elm Det doesn't belong to the authenticated user.")

            if 'UIElementsDetection' in case_study.available_phases:
                return super().get(request, *args, **kwargs)
            else:
                return HttpResponseRedirect(reverse("analyzer:casestudy_list"))
            
        elif 'execution_id' in self.kwargs:
            execution = Execution.objects.get(pk=kwargs["execution_id"])
            if execution.ui_elements_detection:
                return super().get(request, *args, **kwargs)
            else:
                return HttpResponseRedirect(reverse("analyzer:execution_list"))
        
    def get_context_data(self, **kwargs):
        context = super(UIElementsDetectionDetailView, self).get_context_data(**kwargs)
        
        if 'case_study_id' in self.kwargs:
            context['case_study_id'] = self.kwargs['case_study_id']

        if 'execution_id' in self.kwargs:
            #context['execution_id'] = self.kwargs['execution_id']
            context['execution_id'] = self.kwargs['execution_id']
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
                "preloaded": ui_elements_detection.preloaded,
                "preloaded_file": ui_elements_detection.preloaded_file,

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
            if self.object.model == '':
                return

            self.object.user = self.request.user
            self.object.case_study = CaseStudy.objects.get(pk=self.kwargs.get('case_study_id'))
            self.object.save()
            ui_elem_det_obj.ui_elements_classification = self.object
            ui_elem_det_obj.save()
        elif self.object.model == '':
            UIElementsClassification.objects.filter(pk=ui_elem_det_obj.ui_elements_classification.id).delete()
        else:
            UIElementsClassification.objects.filter(pk=ui_elem_det_obj.ui_elements_classification.id).update(**form.cleaned_data)


@login_required(login_url="/login/")
def set_as_ui_elements_detection_active(request):
    ui_elements_detection_id = request.GET.get("ui_elem_detection_id")
    case_study_id = request.GET.get("case_study_id")
    prefilter_list = UIElementsDetection.objects.filter(case_study_id=case_study_id)
    # Validations
    if CaseStudy.objects.get(pk=case_study_id).user != request.user:
        return HttpResponse(status=403, content="Case Study doesn't belong to the authenticated user.")
    if UIElementsDetection.objects.get(pk=ui_elements_detection_id).user != request.user:  
        return HttpResponse(status=403, content="UI Element Detection doesn't belong to the authenticated user.")
    if UIElementsDetection.objects.get(pk=ui_elements_detection_id).case_study != CaseStudy.objects.get(pk=case_study_id):
        return HttpResponse(status=403, content="UI Element Detection doesn't belong to the Case Study.")
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

@login_required(login_url="/login/")
def set_as_ui_elements_detection_inactive(request):
    ui_elements_detection_id = request.GET.get("ui_elem_detection_id")
    case_study_id = request.GET.get("case_study_id")
    # Validations
    if CaseStudy.objects.get(pk=case_study_id).user != request.user:
        return HttpResponse(status=403, content="Case Study doesn't belong to the authenticated user.")
    if UIElementsDetection.objects.get(pk=ui_elements_detection_id).user != request.user:  
        return HttpResponse(status=403, content="UI Element Detection doesn't belong to the authenticated user.")
    if UIElementsDetection.objects.get(pk=ui_elements_detection_id).case_study != CaseStudy.objects.get(pk=case_study_id):
        return HttpResponse(status=403, content="UI Element Detection doesn't belong to the Case Study.")
    ui_elements_detection = UIElementsDetection.objects.get(id=ui_elements_detection_id)
    ui_elements_detection.active = False
    ui_elements_detection.save()
    return HttpResponseRedirect(reverse("featureextraction:ui_detection_list", args=[case_study_id]))
    
@login_required(login_url="/login/")
def delete_ui_elements_detection(request):
    ui_element_detection_id = request.GET.get("ui_elem_detection_id")
    case_study_id = request.GET.get("case_study_id")
    ui_elements_detection = UIElementsDetection.objects.get(id=ui_element_detection_id)
    if request.user.id != ui_elements_detection.user.id:
        raise Exception("This object doesn't belong to the authenticated user")
    ui_elements_detection.delete()
    return HttpResponseRedirect(reverse("featureextraction:ui_detection_list", args=[case_study_id]))


class PrefiltersCreateView(CreateView, LoginRequiredMixin):
    login_url = "/login/"
    model = Prefilters
    form_class = PrefiltersForm
    template_name = "prefiltering/create.html"

    # Check if the the phase can be interacted with (included in case study available phases)
    def get(self, request, *args, **kwargs):
        case_study = CaseStudy.objects.get(pk=kwargs["case_study_id"])
        if not case_study:
            return HttpResponse(status=404, content="Prefilter not found.")
        elif case_study.user != request.user:
            return HttpResponse(status=403, content="Prefilter doesn't belong to the authenticated user.")

        if 'Prefilters' in case_study.available_phases:
            return super().get(request, *args, **kwargs)
        else:
            return HttpResponseRedirect(reverse("analyzer:casestudy_list"))
    
    def post(self, request, *args, **kwargs):
        case_study = CaseStudy.objects.get(pk=kwargs["case_study_id"])
        if not case_study:
            return HttpResponse(status=404, content="Prefilter not found.")
        elif case_study.user != request.user:
            return HttpResponse(status=403, content="Prefilter doesn't belong to the authenticated user.")

        if 'Prefilters' in case_study.available_phases:
            return super().post(request, *args, **kwargs)
        else:
            return HttpResponseRedirect(reverse("analyzer:casestudy_list"))
    
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

class PrefiltersListView(ListView, LoginRequiredMixin):
    login_url = "/login/"
    model = Prefilters
    template_name = "prefiltering/list.html"
    paginate_by = 50

    # Check if the the phase can be interacted with (included in case study available phases)
    def get(self, request, *args, **kwargs):
        case_study = CaseStudy.objects.get(pk=kwargs["case_study_id"])
        if not case_study:
            return HttpResponse(status=404, content="Prefilter not found.")
        elif case_study.user != request.user:
            return HttpResponse(status=403, content="Prefilter doesn't belong to the authenticated user.")

        if 'Prefilters' in case_study.available_phases:
            return super().get(request, *args, **kwargs)
        else:
            return HttpResponseRedirect(reverse("analyzer:casestudy_list"))

    def get_context_data(self, **kwargs):
        context = super(PrefiltersListView, self).get_context_data(**kwargs)
        context['case_study_id'] = self.kwargs.get('case_study_id')
        return context

    def get_queryset(self):
        # Obtiene el ID del Experiment pasado como parámetro en la URL
        case_study_id = self.kwargs.get('case_study_id')

        # Search if s is a query parameter
        search = self.request.GET.get("s")
        # Filtra los objetos por case_study_id
        if search:
            queryset = Prefilters.objects.filter(case_study__id=case_study_id, case_study__user=self.request.user, title__icontains=search).order_by('-created_at')
        else:
            queryset = Prefilters.objects.filter(case_study__id=case_study_id, case_study__user=self.request.user).order_by('-created_at')

        return queryset

    
class PrefiltersDetailView(DetailView, LoginRequiredMixin):
    login_url = "/login/"
    # Check if the the phase can be interacted with (included in case study available phases)
    
    def get(self, request, *args, **kwargs):
        prefilter = get_object_or_404(Prefilters, id=kwargs["prefilter_id"])
        if prefilter.case_study.user != request.user:
            return HttpResponse(status=403, content="Prefilter doesn't belong to the authenticated user.")

        form = PrefiltersForm(read_only=True, instance=prefilter)
        if 'case_study_id' in kwargs:
            case_study = get_object_or_404(CaseStudy, id=kwargs['case_study_id'])
            if 'Prefilters' in case_study.available_phases:
                context= {"prefilter": prefilter, 
                    "case_study_id": case_study.id,
                    "form": form,}
        
                return render(request, "prefiltering/detail.html", context)
            else:
                return HttpResponseRedirect(reverse("analyzer:casestudy_list"))
         
        elif 'execution_id' in kwargs:
            execution = get_object_or_404(Execution, id=kwargs['execution_id'])
            if execution.prefilters:
                context= {"prefilter": prefilter, 
                            "execution_id": execution.id,
                            "form": form,}
            
                return render(request, "prefiltering/detail.html", context)
            else:
                return HttpResponseRedirect(reverse("analyzer:execution_list"))
    
@login_required(login_url="/login/")   
def set_as_prefilters_active(request):
    prefilter_id = request.GET.get("prefilter_id")
    case_study_id = request.GET.get("case_study_id")
    if not CaseStudy.objects.get(pk=case_study_id):
        return HttpResponse(status=404, content="Case Study not found.")
    elif not CaseStudy.objects.get(pk=case_study_id).user == request.user:
        return HttpResponse(status=403, content="Case Study doesn't belong to the authenticated user.")
    prefilter_list = Prefilters.objects.filter(case_study_id=case_study_id)
    for m in prefilter_list:
        m.active = False
        m.save()
    prefilter = Prefilters.objects.get(id=prefilter_id)
    prefilter.active = True
    prefilter.save()
    return HttpResponseRedirect(reverse("featureextraction:prefilters_list", args=[case_study_id]))

@login_required(login_url="/login/")   
def set_as_prefilters_inactive(request):
    prefilter_id = request.GET.get("prefilter_id")
    case_study_id = request.GET.get("case_study_id")
    if not CaseStudy.objects.get(pk=case_study_id):
        return HttpResponse(status=404, content="Case Study not found.")
    elif not CaseStudy.objects.get(pk=case_study_id).user == request.user:
        return HttpResponse(status=403, content="Case Study doesn't belong to the authenticated user.")

    prefilter = Prefilters.objects.get(id=prefilter_id)
    prefilter.active = False
    prefilter.save()
    return HttpResponseRedirect(reverse("featureextraction:prefilters_list", args=[case_study_id]))
    
@login_required(login_url="/login/")   
def delete_prefilter(request):
    prefilter_id = request.GET.get("prefilter_id")
    case_study_id = request.GET.get("case_study_id")
    if not CaseStudy.objects.get(pk=case_study_id):
        return HttpResponse(status=404, content="Case Study not found.")
    elif not CaseStudy.objects.get(pk=case_study_id).user == request.user:
        return HttpResponse(status=403, content="Case Study doesn't belong to the authenticated user.")
    prefilter = Prefilters.objects.get(id=prefilter_id)
    prefilter.delete()
    return HttpResponseRedirect(reverse("featureextraction:prefilters_list", args=[case_study_id]))

class PostfiltersCreateView(CreateView, LoginRequiredMixin):
    login_url = "/login/"
    model = Postfilters
    form_class = PostfiltersForm
    template_name = "postfiltering/create.html"

    # Check if the the phase can be interacted with (included in case study available phases)
    def get(self, request, *args, **kwargs):
        case_study = CaseStudy.objects.get(pk=kwargs["case_study_id"])
        if not case_study:
            return HttpResponse(status=404, content="Postfilters not found.")
        elif case_study.user != request.user:
            return HttpResponse(status=403, content="Postfilters doesn't belong to the authenticated user.")

        if 'Postfilters' in case_study.available_phases:
            return super().get(request, *args, **kwargs)
        else:
            return HttpResponseRedirect(reverse("analyzer:casestudy_list"))
    
    def post(self, request, *args, **kwargs):
        case_study = CaseStudy.objects.get(pk=kwargs["case_study_id"])
        if not case_study:
            return HttpResponse(status=404, content="Postfilters not found.")
        elif case_study.user != request.user:
            return HttpResponse(status=403, content="Postfilters doesn't belong to the authenticated user.")

        if 'Postfilters' in case_study.available_phases:
            return super().post(request, *args, **kwargs)
        else:
            return HttpResponseRedirect(reverse("analyzer:casestudy_list"))
    
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

class PostfiltersListView(ListView, LoginRequiredMixin):
    login_url = "/login/"
    model = Postfilters
    template_name = "postfiltering/list.html"
    paginate_by = 50

    # Check if the the phase can be interacted with (included in case study available phases)
    def get(self, request, *args, **kwargs):
        case_study = CaseStudy.objects.get(pk=kwargs["case_study_id"])
        if not case_study:
            return HttpResponse(status=404, content="Postfilters not found.")
        elif case_study.user != request.user:
            return HttpResponse(status=403, content="Postfilters doesn't belong to the authenticated user.")

        if 'Postfilters' in case_study.available_phases:
            return super().get(request, *args, **kwargs)
        else:
            return HttpResponseRedirect(reverse("analyzer:casestudy_list"))

    def get_context_data(self, **kwargs):
        context = super(PostfiltersListView, self).get_context_data(**kwargs)
        context['case_study_id'] = self.kwargs.get('case_study_id')
        return context

    def get_queryset(self):
        # Obtiene el ID del Experiment pasado como parámetro en la URL
        case_study_id = self.kwargs.get('case_study_id')

        # Search if s is a query parameter
        search = self.request.GET.get("s")
        # Filtra los objetos por case_study_id
        if search:
            queryset = Postfilters.objects.filter(case_study__id=case_study_id, case_study__user=self.request.user, title__icontains=search).order_by('-created_at')
        else:
            queryset = Postfilters.objects.filter(case_study__id=case_study_id, case_study__user=self.request.user).order_by('-created_at')

        return queryset
    
class PostfiltersDetailView(DetailView, LoginRequiredMixin):
    login_url = "/login/"
    # Check if the the phase can be interacted with (included in case study available phases)
    def get(self, request, *args, **kwargs):
        form = PostfiltersForm(read_only=True, instance=postfilter)
        postfilter = get_object_or_404(Postfilters, id=kwargs["postfilter_id"])
        if not postfilter:
            return HttpResponse(status=404, content="Postfilters not found.")
        elif postfilter.case_study.user != request.user:
            return HttpResponse(status=403, content="Postfilters doesn't belong to the authenticated user.")

        if 'case_study_id' in kwargs:
            case_study = get_object_or_404(CaseStudy, id=kwargs['case_study_id'])
            if 'Postfilters' in case_study.available_phases:
                postfilter = get_object_or_404(Postfilters, id=kwargs["postfilter_id"])
                context= {"postfilter": postfilter, 
                    "case_study_id": case_study.id,
                    "form": form,}
        
                return render(request, "postfiltering/detail.html", context)
            else:
                return HttpResponseRedirect(reverse("analyzer:casestudy_list"))
         
        elif 'execution_id' in kwargs:
            execution = get_object_or_404(Execution, id=kwargs['execution_id'])
            if execution.postfilters:
                context= {"postfilter": postfilter, 
                            "execution_id": execution.id,
                            "form": form,}
            
                return render(request, "postfiltering/detail.html", context)
            else:
                return HttpResponseRedirect(reverse("analyzer:execution_list"))
    
@login_required(login_url="/login/")   
def set_as_postfilters_active(request):
    postfilter_id = request.GET.get("postfilter_id")
    case_study_id = request.GET.get("case_study_id")
    if not CaseStudy.objects.get(pk=case_study_id):
        return HttpResponse(status=404, content="Case Study not found.")
    elif not CaseStudy.objects.get(pk=case_study_id).user == request.user:
        return HttpResponse(status=403, content="Case Study doesn't belong to the authenticated user.")
    postfilter_list = Postfilters.objects.filter(case_study_id=case_study_id)
    for m in postfilter_list:
        m.active = False
        m.save()
    postfilter = Postfilters.objects.get(id=postfilter_id)
    postfilter.active = True
    postfilter.save()
    return HttpResponseRedirect(reverse("featureextraction:postfilters_list", args=[case_study_id]))

@login_required(login_url="/login/")
def set_as_postfilters_inactive(request):
    postfilter_id = request.GET.get("postfilter_id")
    case_study_id = request.GET.get("case_study_id")
    # Validations
    if not request.user.is_authenticated:
        raise ValidationError(_("User must be authenticated."))
    if CaseStudy.objects.get(pk=case_study_id).user != request.user:
        raise ValidationError(_("Case Study doesn't belong to the authenticated user."))
    if Postfilters.objects.get(pk=postfilter_id).user != request.user:  
        raise ValidationError(_("Postfiltering doesn't belong to the authenticated user."))
    if Postfilters.objects.get(pk=postfilter_id).case_study != CaseStudy.objects.get(pk=case_study_id):
        raise ValidationError(_("Postfiltering doesn't belong to the Case Study."))   
    postfilter = Postfilters.objects.get(id=postfilter_id)
    postfilter.active = False
    postfilter.save()
    return HttpResponseRedirect(reverse("featureextraction:postfilters_list", args=[case_study_id]))
    
@login_required(login_url="/login/")
def delete_postfilter(request):
    postfilter_id = request.GET.get("postfilter_id")
    # Validations
    if not request.user.is_authenticated:
        raise ValidationError(_("User must be authenticated."))
    if CaseStudy.objects.get(pk=case_study_id).user != request.user:
        raise ValidationError(_("Case Study doesn't belong to the authenticated user."))
    if Postfilters.objects.get(pk=postfilter_id).user != request.user:  
        raise ValidationError(_("Postfiltering doesn't belong to the authenticated user."))
    if Postfilters.objects.get(pk=postfilter_id).case_study != CaseStudy.objects.get(pk=case_study_id):
        raise ValidationError(_("Postfiltering doesn't belong to the Case Study."))   
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
                draw_postfilter_relevant_ui_compos_borders(os.path.join(execution.exp_folder_complete_path, scenario))

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
                draw_ui_compos_borders(os.path.join(execution.exp_folder_complete_path ,scenario))

            # else:
            #     raise Exception("You don't have permissions to access this files")
            response = 'Postfiltered UI compos borders has been drawn!'
        else:
            response = 'The processing of this case study has not yet finished, please try again in a few minutes'

    except Exception as e:
        response = f"Case Study with id {execution_id} raise an exception: " + str(e)
        st = status.HTTP_404_NOT_FOUND

    return HttpResponse(response, status=st)


##########################################################

class UIElementsDetectionResultDetailView(LoginRequiredMixin, DetailView):
    login_url = "/login/"

    def get(self, request, *args, **kwargs):
        user = request.user
        execution: Execution = get_object_or_404(Execution, id=kwargs["execution_id"])     
        if user.id != execution.user.id:
            return HttpResponse(status=403, content=_("Execution doesn't belong to the authenticated user."))
        scenario: str = request.GET.get('scenario')
        download = request.GET.get('download')

        if scenario == None:
            scenario = execution.scenarios_to_study[0] # Select the first scenario by default

        # Create dictionary with images and their corresponding UI elements
        soms = dict()

        classes = execution.ui_elements_classification.model.classes
        colors = []
        for i in range(len(classes)):
            colors.append("#%06x" % random.randint(0, 0xFFFFFF))
        soms["classes"] = {k: v for k, v in zip(classes, colors)} 

        soms["soms"] = []

        for compo_json in os.listdir(os.path.join(execution.exp_folder_complete_path, scenario + "_results", "components_json")):
            with open(os.path.join(execution.exp_folder_complete_path, scenario + "_results", "components_json", compo_json), "r") as f:
                compos = json.load(f)
            # path is something like: asdsa/.../.../image.PNG.json
            img_name = compo_json.split("/")[-1].split(".json")[0]
            img_path = os.path.join(execution.case_study.exp_foldername, scenario, img_name)

            soms["soms"].append(
                {
                    "img": img_name,
                    "img_path": img_path,
                    "som": compos
                }
            )

        context = {
            "execution_id": execution.id,
            "scenarios": execution.scenarios_to_study,
            "soms": soms
        }

        #return HttpResponse(json.dumps(context), content_type="application/json")
        return render(request, "ui_elements_detection/results.html", context)
