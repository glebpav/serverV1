
from django.urls import path, include

from .views import *

urlpatterns = [
    path('', hello),
    path('themFind', find),
]
