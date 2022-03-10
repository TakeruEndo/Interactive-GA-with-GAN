from django.urls import path
from django.conf.urls import url
from . import views

urlpatterns = [
    path('', views.index, name='index'),
    # ex: /polls/5/
    path('next_generation/<int:new_generation>',
         views.next_generation, name='next_generation'),
    path('re_create', views.re_create, name='re_create'),
]
