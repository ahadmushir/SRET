from django.urls import path

from . import views

urlpatterns = [
    path('', views.index, name='index'),
    path('temp/', views.temp, name='temp'),
    path('ssh_model/', views.ssh_model, name='ssh_model'),
    path('ssh_model_body/', views.ssh_model_body, name='ssh_model_body'),
    path('display_results/', views.display_results, name='display_results'),

]