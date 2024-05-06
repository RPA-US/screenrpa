from django.http import HttpResponseRedirect
from django.shortcuts import render, get_object_or_404
from django.urls import reverse
from django.views.generic import ListView, CreateView, UpdateView, DetailView
from django.core.exceptions import ValidationError
from .models import Monitoring
from apps.analyzer.models import CaseStudy
from .forms import MonitoringForm
from django.utils.translation import gettext_lazy as _
from django.contrib.auth.mixins import LoginRequiredMixin

# Create your views here.
    
class MonitoringCreateView(CreateView, LoginRequiredMixin):
    model = Monitoring
    form_class = MonitoringForm
    template_name = "monitoring/create.html"
    login_url = '/login/'
    
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

class MonitoringListView(ListView, LoginRequiredMixin):
    model = Monitoring
    template_name = "monitoring/list.html"
    paginate_by = 50

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

class MonitoringDetailView(DetailView):
    def get(self, request, *args, **kwargs):
        monitoring = get_object_or_404(Monitoring, id=kwargs["monitoring_id"])
        return render(request, "monitoring/detail.html", {"monitoring": monitoring})
    

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
    
def delete_monitoring(request):
    monitoring_id = request.GET.get("monitoring_id")
    case_study_id = request.GET.get("case_study_id")
    monitoring = Monitoring.objects.get(id=monitoring_id)
    if request.user.id != monitoring.user.id:
        raise Exception(_("This object doesn't belong to the authenticated user"))
    monitoring.delete()
    return HttpResponseRedirect(reverse("behaviourmonitoring:monitoring_list", args=[case_study_id]))

