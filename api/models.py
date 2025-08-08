from django.db import models
from django.contrib.postgres.fields import ArrayField
import numpy as np


class PhysicsConcept(models.Model):
    """Store physics concepts with embeddings"""
    name = models.CharField(max_length=200, unique=True)
    description = models.TextField()
    embedding = models.JSONField(null=True)  # Store embedding vector
    field = models.CharField(max_length=100)  # Physics field
    difficulty_level = models.IntegerField(default=1)
    example_equations = models.JSONField(default=list)

    def __str__(self):
        return self.name


class MITCourse(models.Model):
    """MIT OpenCourseWare courses with embeddings"""
    course_number = models.CharField(max_length=20, unique=True)
    title = models.CharField(max_length=300)
    description = models.TextField()
    syllabus = models.TextField(null=True)
    url = models.URLField()
    department = models.CharField(max_length=100)
    level = models.CharField(max_length=50)

    # Embeddings for semantic search
    title_embedding = models.JSONField(null=True)
    description_embedding = models.JSONField(null=True)
    syllabus_embedding = models.JSONField(null=True)

    # Course content
    topics_covered = models.JSONField(default=list)
    equations_covered = models.JSONField(default=list)
    prerequisites = models.ManyToManyField('self', symmetrical=False, related_name='required_for')
    concepts = models.ManyToManyField(PhysicsConcept, related_name='courses')

    def __str__(self):
        return f"{self.course_number}: {self.title}"


class ConceptEmbedding(models.Model):
    """Store precomputed embeddings for concepts"""
    text = models.TextField(unique=True)
    embedding_scibert = models.JSONField()
    embedding_mathbert = models.JSONField()
    created_at = models.DateTimeField(auto_now_add=True)


class AnalysisRequest(models.Model):
    """Enhanced analysis requests with embeddings"""
    url = models.URLField()
    extracted_text = models.TextField()

    # Extracted features
    identified_concepts = models.JSONField(default=list)
    extracted_equations = models.JSONField(default=list)
    concept_embeddings = models.JSONField(default=list)

    # Recommendations
    recommended_courses = models.JSONField(default=list)
    confidence_scores = models.JSONField(default=dict)

    created_at = models.DateTimeField(auto_now_add=True)