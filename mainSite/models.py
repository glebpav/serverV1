from django.db import models
import json


class News(models.Model):
    title = models.CharField(max_length=30)
    body = models.CharField(max_length=1000)
    url = models.CharField(max_length=100)
    rating = models.CharField(max_length=10)

    def __str__(self):
        return self.title


class Person(models.Model):
    id = models.AutoField(primary_key=True)
    login = models.CharField(max_length=30)
    name = models.CharField(max_length=30)
    password = models.CharField(max_length=30)
    themes = models.CharField(max_length=1000, default=None, blank=True, null=True)
    history = models.CharField(max_length=1000, default=None, blank=True, null=True)

    def __str__(self):
        return self.name
