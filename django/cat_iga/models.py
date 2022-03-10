import datetime

from django.db import models
from django.utils import timezone


class User(models.Model):
    name = models.CharField(max_length=32)
    mail = models.EmailField()


class Entry(models.Model):
    STATUS_DRAFT = "draft"
    STATUS_PUBLIC = "public"
    title = models.CharField(max_length=128)
    body = models.TextField()
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    author = models.ForeignKey(
        User, related_name='entries', on_delete=models.CASCADE)
