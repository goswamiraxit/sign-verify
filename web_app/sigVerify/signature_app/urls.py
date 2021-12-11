from django.urls import path
from django.contrib import admin
from . import views


urlpatterns = [
    path('', views.home, name='home'),
    path('customer/', views.CustomerView.as_view(), name='customer'),
    path('customer/<str:pk>', views.CustomerSigVer.as_view(), name='cust_sig_ver'),
]

