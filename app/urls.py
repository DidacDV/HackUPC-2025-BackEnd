from django.urls import path
from . import views

urlpatterns = [
    path('api/<str:nombre_csv>/', views.mostrar_csv_dinamico, name='mostrar_csv_dinamico'),
]
