import csv
import json
import os
from django.db.models.query import QuerySet
from django.http import HttpRequest, HttpResponse, HttpResponseRedirect
from django.shortcuts import render, get_object_or_404
from django.urls import reverse
from django.views.generic import ListView, CreateView, UpdateView, DetailView
from django.core.exceptions import ValidationError, PermissionDenied
from .models import Monitoring
from apps.analyzer.models import CaseStudy, Execution
from .forms import MonitoringForm
from django.utils.translation import gettext_lazy as _
from django.contrib.auth.mixins import LoginRequiredMixin
from django.contrib.auth.decorators import login_required
from core.utils import read_ui_log_as_dataframe

# Create your views here.
    
class MonitoringCreateView(LoginRequiredMixin, CreateView):
    model = Monitoring
    form_class = MonitoringForm
    template_name = "monitoring/create.html"
    login_url = '/login/'

    def get(self, request, *args, **kwargs):
        case_study = get_object_or_404(CaseStudy, id=kwargs['case_study_id'])
        return super().get(request, *args, **kwargs) 
    
    def get_context_data(self, **kwargs):
        context = super(MonitoringCreateView, self).get_context_data(**kwargs)
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

class MonitoringListView(LoginRequiredMixin, ListView):
    login_url = '/login/'
    model = Monitoring
    template_name = "monitoring/list.html"
    paginate_by = 50

    def get(self, request, *args, **kwargs) -> HttpResponse:
        case_study = get_object_or_404(CaseStudy, id=kwargs['case_study_id'])
        if case_study.user != request.user:
            raise PermissionDenied("This object doesn't belong to the authenticated user")
        return super().get(request, *args, **kwargs)

    def get_context_data(self, **kwargs):
        context = super(MonitoringListView, self).get_context_data(**kwargs)
        context['case_study_id'] = self.kwargs.get('case_study_id')
        return context

    def get_queryset(self):
        # Obtiene el ID del Experiment pasado como parámetro en la URL
        case_study_id = self.kwargs.get('case_study_id')

        # Search if s is a query parameter
        search = self.request.GET.get("s")
        # Filtra los objetos Monitoring por case_study_id
        if search:
            queryset = Monitoring.objects.filter(case_study__id=case_study_id, case_study__user=self.request.user, title__icontains=search).order_by('-created_at')
        else:
            queryset = Monitoring.objects.filter(case_study__id=case_study_id, case_study__user=self.request.user).order_by('-created_at')
    
        return queryset

class MonitoringDetailView(LoginRequiredMixin, UpdateView):
    login_url = '/login/'
    model = Monitoring
    success_url = "/monitoring/list/"
    form_class = MonitoringForm

    def get_object(self, queryset=None):
        monitoring_id = self.kwargs.get('monitoring_id')
        
        return get_object_or_404(Monitoring, id=monitoring_id)

    def get(self, request, *args, **kwargs):
        case_study = get_object_or_404(CaseStudy, id=kwargs["case_study_id"])
        monitoring = self.get_object()
        
        if not case_study.user == request.user:
            raise PermissionDenied("This object doesn't belong to the authenticated user")

        form = MonitoringForm(read_only=monitoring.freeze, instance=monitoring)

        context={}

        if 'case_study_id' in kwargs:
            case_study = get_object_or_404(CaseStudy, id=kwargs['case_study_id'])

            context= {"monitoring": monitoring, 
                  "case_study_id": case_study.id,
                  "form": form,}

        elif 'execution_id' in kwargs:
            execution = get_object_or_404(Execution, id=kwargs['execution_id'])

            context= {"monitoring": monitoring, 
                        "execution_id": execution.id,
                        "form": form,}
        
        return render(request, "monitoring/detail.html", context)

    def form_valid(self, form, *args, **kwargs):
        if not self.request.user.is_authenticated:
            raise ValidationError("User must be authenticated.")
        if self.object.freeze:
            raise ValidationError("This object cannot be edited.")
        if not self.object.case_study.user == self.request.user:
            raise PermissionDenied("This object doesn't belong to the authenticated")
        self.object.save()
        return HttpResponseRedirect(self.get_success_url() + str(self.object.case_study.id))
    

@login_required(login_url='/login/')
def set_as_active(request):
    monitoring_id = request.GET.get("monitoring_id")
    case_study_id = request.GET.get("case_study_id")
    
    # Validations
    if not request.user.is_authenticated:
        raise ValidationError(_("User must be authenticated."))
    if CaseStudy.objects.get(pk=case_study_id).user != request.user:
        raise ValidationError(_("Case Study doesn't belong to the authenticated user."))
    if Monitoring.objects.get(pk=monitoring_id).user != request.user:  
        raise ValidationError(_("Monitoring doesn't belong to the authenticated user."))
    if Monitoring.objects.get(pk=monitoring_id).case_study != CaseStudy.objects.get(pk=case_study_id):
        raise ValidationError(_("Monitoring doesn't belong to the Case Study."))
    
    monitoring_list = Monitoring.objects.filter(case_study_id=case_study_id, active=True)
    for m in monitoring_list:
        m.active = False
        m.save()
    monitoring = Monitoring.objects.get(id=monitoring_id)
    monitoring.active = True
    monitoring.save()
    return HttpResponseRedirect(reverse("behaviourmonitoring:monitoring_list", args=[case_study_id]))

@login_required(login_url='/login/')
def set_as_inactive(request):
    monitoring_id = request.GET.get("monitoring_id")
    case_study_id = request.GET.get("case_study_id")
    # Validations
    if not request.user.is_authenticated:
        raise ValidationError(_("User must be authenticated."))
    if CaseStudy.objects.get(pk=case_study_id).user != request.user:
        raise ValidationError(_("Case Study doesn't belong to the authenticated user."))
    if Monitoring.objects.get(pk=monitoring_id).user != request.user:  
        raise ValidationError(_("Monitoring doesn't belong to the authenticated user."))
    if Monitoring.objects.get(pk=monitoring_id).case_study != CaseStudy.objects.get(pk=case_study_id):
        raise ValidationError(_("Monitoring doesn't belong to the Case Study."))
    monitoring = Monitoring.objects.get(id=monitoring_id)
    monitoring.active = False
    monitoring.save()
    return HttpResponseRedirect(reverse("behaviourmonitoring:monitoring_list", args=[case_study_id]))
    
@login_required(login_url='/login/')
def delete_monitoring(request):
    monitoring_id = request.GET.get("monitoring_id")
    case_study_id = request.GET.get("case_study_id")
    if not CaseStudy.objects.get(pk=case_study_id):
        return HttpResponse(status=404, content="Case Study not found")
    elif not CaseStudy.objects.get(pk=case_study_id).user == request.user:
        raise PermissionDenied("This object doesn't belong to the authenticated user")
    monitoring = Monitoring.objects.get(id=monitoring_id)
    if request.user.id != monitoring.user.id:
        raise PermissionDenied(_("This object doesn't belong to the authenticated user"))
    monitoring.delete()
    return HttpResponseRedirect(reverse("behaviourmonitoring:monitoring_list", args=[case_study_id]))

###########################################################

class MonitoringResultDetailView(LoginRequiredMixin, DetailView):
    login_url = '/login/'
    def get(self, request, *args, **kwargs):
        # Get the Execution object or raise a 404 error if not found
        execution = get_object_or_404(Execution, id=kwargs["execution_id"])     
        if not execution.case_study.user == request.user:
            raise PermissionDenied("This object doesn't belong to the authenticated user")
        scenario = request.GET.get('scenario')
        download = request.GET.get('download')

        if scenario == None:
            #scenario = "1"
            scenario = execution.scenarios_to_study[0] # by default, the first one that was indicated
            
        #path_to_csv_file = execution.exp_folder_complete_path + "/"+ scenario +"/log.csv"  
        path_to_csv_file = os.path.join(execution.exp_folder_complete_path, scenario, "log.csv")
        # CSV Download
        if path_to_csv_file and download=="True":
            return ResultDownload(path_to_csv_file)  

        # CSV Reading and Conversion to JSON
        csv_data_json = read_ui_log_as_dataframe(path_to_csv_file, lib='polars').to_dicts()

        # Include CSV data in the context for the template
        context = {
            "execution_id": execution.id,
            "csv_data": csv_data_json,  # Data to be used in the HTML template
            "scenarios": execution.scenarios_to_study,
            "scenario": scenario
            } 

        # Render the HTML template with the context including the CSV data
        return render(request, "monitoring/result.html", context)


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