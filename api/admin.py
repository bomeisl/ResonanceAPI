from django.contrib import admin
from .models import PhysicsConcept, MITCourse, AnalysisRequest

@admin.register(PhysicsConcept)
class PhysicsConceptAdmin(admin.ModelAdmin):
    list_display = ['name', 'field', 'difficulty_level']
    search_fields = ['name', 'description']
    list_filter = ['field', 'difficulty_level']

@admin.register(MITCourse)
class MITCourseAdmin(admin.ModelAdmin):
    list_display = ['course_number', 'title', 'level', 'department']
    search_fields = ['course_number', 'title', 'description']
    list_filter = ['level', 'department']
    filter_horizontal = ['prerequisites', 'concepts']

@admin.register(AnalysisRequest)
class AnalysisRequestAdmin(admin.ModelAdmin):
    list_display = ['url', 'created_at']
    list_filter = ['created_at']
    readonly_fields = ['created_at']