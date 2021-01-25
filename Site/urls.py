
from django.contrib import admin
from django.urls import path, include

urlpatterns = [
    path('', include('mainSite.urls')),
    path('admin/', admin.site.urls),
    path('api/',include('mainSite.urls')),
]
