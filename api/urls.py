from django.urls import path
from . import views

urlpatterns = [
    path('health/', views.health_check, name='health-check'),
    path('analyze/', views.analyze_physics_text, name='analyze-physics'),
    path('courses/', views.list_courses, name='list-courses'),
]