from django.urls import path
from . import views

urlpatterns = [
    path('', views.home, name='home'),
    path('api/nfc/', views.receive_nfc_data, name='receive_nfc_data'),
     path('api/transfer/', views.transfer_xcoin),
     path('api/balance/', views.balance_xcoin),
]
