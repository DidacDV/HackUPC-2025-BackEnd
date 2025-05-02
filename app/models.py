from django.db import models

class Building(models.Model):
    name = models.CharField(max_length=100)
    consumption = models.FloatField()
    created_at = models.DateTimeField(auto_now_add=True)

class Test1(models.Model):
    name = models.CharField(max_length=100)
