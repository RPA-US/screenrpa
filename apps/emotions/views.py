from django.shortcuts import render

# Create your views here.

def emotion_home(request):
    """
    Render the emotion home page.
    """
    return render(request, 'emotions/emotion_home.html')
