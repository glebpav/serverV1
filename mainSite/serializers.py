from rest_framework import serializers

from .models import *


class NewsSerializer(serializers.Serializer):
    title = serializers.CharField(max_length=30)
    body = serializers.CharField(max_length=1000)
    url = serializers.CharField(max_length=100)
    rating = serializers.CharField(max_length=10)


class UserSerializer(serializers.Serializer):
    id = serializers.PrimaryKeyRelatedField(read_only=True)
    login = serializers.CharField(max_length=30)
    name = serializers.CharField(max_length=30)
    password = serializers.CharField(max_length=30)
    themes = serializers.CharField(max_length=1000, default=None, allow_blank=True)
    history = serializers.CharField(max_length=1000, default=None, allow_blank=True)

    def create(self, validated_data):
        return Person.objects.create(**validated_data)

    def update(self, instance, validated_data):
        instance.login = validated_data.get('login', instance.login)
        instance.name = validated_data.get('name', instance.name)
        instance.password = validated_data.get('password', instance.password)
        instance.themes = validated_data.get('themes', instance.themes)
        instance.history = validated_data.get('history', instance.history)
        instance.save()
        return instance


class UserCheckerSerializer(serializers.Serializer):
    login = serializers.CharField(max_length=30)
    password = serializers.CharField(max_length=30)