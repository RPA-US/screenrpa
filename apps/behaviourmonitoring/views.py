from django.http import HttpResponseRedirect
from django.views.generic import ListView, CreateView
from django.core.exceptions import ValidationError
from .models import GazeAnalysis
from .forms import GazeAnalysisForm

# Create your views here.
    
class GazeAnalysisCreateView(CreateView):
    model = GazeAnalysis
    form_class = GazeAnalysisForm
    template_name = "gaze_analysis/create.html"

    def form_valid(self, form):
        if not self.request.user.is_authenticated:
            raise ValidationError("User must be authenticated.")
        self.object = form.save(commit=False)
        self.object.user = self.request.user
        saved = self.object.save()
        return HttpResponseRedirect(self.get_success_url())

class GazeAnalysisListView(ListView):
    model = GazeAnalysis
    template_name = "gaze_analysis/list.html"
    paginate_by = 50

    def get_queryset(self):
        return GazeAnalysis.objects.all()
