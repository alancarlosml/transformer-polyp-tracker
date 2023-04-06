from django.http import HttpResponse
from django.shortcuts import render
from django.views.generic import ListView, DetailView
from django.views.generic.edit import CreateView, UpdateView, DeleteView
from django.urls import reverse_lazy

from .models import VideoProcessing

# Create your views here.

class VideoProcessingList(ListView):
    model = VideoProcessing

class VideoProcessingDetail(DetailView):
    model = VideoProcessing