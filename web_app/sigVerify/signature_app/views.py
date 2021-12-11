from django.shortcuts import render

# Create your views here.
from .models import customer

def home(request) :
    """" View Function for the home page"""

    # Generate the count of objects in DataBase
    num_cust = customer.objects.all().count()

    context = {
        'num_cust': num_cust,
    }
    # Render the HTML Template home.thml with data in the context variable
    return render(request, 'home.html', context=context)

from django.views import generic

class CustomerView(generic.ListView) :
    model = customer

class CustomerSigVer(generic.DetailView) :
    model = customer