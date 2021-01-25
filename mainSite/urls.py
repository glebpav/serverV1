
from django.urls import path, include

from .views import *

urlpatterns = [
    path('', hello),
    path('themFind/', find),
    path('news/', News_view.as_view()),
    path('news/<theme>/', News_view.as_view()),
    path('user_check/', User_check.as_view()),
    path('users/<int:pk>', User_view.as_view()),
    path('users/', User_view.as_view()),
]
