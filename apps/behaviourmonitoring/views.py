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
        return Monitoring.objects.all()
