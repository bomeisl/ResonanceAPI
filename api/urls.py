from django.urls import path
from . import views

urlpatterns = [
    path('health/', views.health_check, name='health-check'),
    path('analyze/', views.analyze_physics_text, name='analyze-physics'),
    path('analyze-text/', views.analyze_text_direct, name='analyze-text'),
    path('courses/', views.list_courses, name='list-courses'),
    path('test-wikipedia/', views.test_wikipedia, name='test-wikipedia'),
]