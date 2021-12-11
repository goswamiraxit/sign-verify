from django.db import models
from django.urls import reverse

# Create your models here.

class customer(models.Model):
    'Model representing a customer needing signature verification'
    fname = models.CharField('First Name', max_length=20)

    lname = models.CharField('Last Name', max_length=20, primary_key=True)

    signature = models.ImageField(upload_to='customers')

    def __str__(self):
        return f'{self.fname} {self.lname}'

    def get_absolute_url(self):
        return reverse('cust_sig_ver', args=[str(self.lname)])