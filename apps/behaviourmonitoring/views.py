from django.http import HttpResponseRedirect
from django.views.generic import ListView, CreateView
from django.core.exceptions import ValidationError
from .models import Monitoring
from .forms import MonitoringForm

# Create your views here.
    
class MonitoringCreateView(CreateView):
    model = Monitoring
    form_class = MonitoringForm
    template_name = "monitoring/create.html"

    def form_valid(self, form):
        if not self.request.user.is_authenticated:
            raise ValidationError("User must be authenticated.")
        self.object = form.save(commit=False)
        self.object.user = self.request.user
        saved = self.object.save()
        return HttpResponseRedirect(self.get_success_url())

class MonitoringListView(ListView):
    model = Monitoring
    template_name = "monitoring/list.html"
    paginate_by = 50

    def get_queryset(self):
        # Obtiene el ID del Experiment pasado como parámetro en la URL
        case_study_id = self.kwargs.get('case_study_id')

        # Filtra los objetos Monitoring por case_study_id
        queryset = Monitoring.objects.filter(case_study__id=case_study_id, case_study__user=self.request.user).order_by('-created_at')

        return queryset