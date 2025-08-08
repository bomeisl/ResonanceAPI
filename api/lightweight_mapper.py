from typing import List, Dict
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import logging

logger = logging.getLogger(__name__)


class LightweightCourseMapper:
    '''Lightweight course mapper using TF-IDF instead of BERT'''

    def __init__(self):
        self.courses = self._initialize_courses()
        self.vectorizer = TfidfVectorizer(max_features=500, stop_words='english')
        self._prepare_course_vectors()

    def _initialize_courses(self):
        '''Initialize MIT courses database'''
        return {
            "8.01": {
                "title": "Physics I: Classical Mechanics",
                "description": "Newtonian mechanics forces momentum energy kinematics dynamics",
                "level": "Undergraduate",
                "prerequisites": [],
                "url": "https://ocw.mit.edu/courses/8-01sc-classical-mechanics-fall-2016/"
            },
            "8.02": {
                "title": "Physics II: Electricity and Magnetism",
                "description": "Electric magnetic fields Maxwell equations electromagnetic waves",
                "level": "Undergraduate",
                "prerequisites": ["8.01"],
                "url": "https://ocw.mit.edu/courses/8-02-physics-ii-electricity-and-magnetism-spring-2007/"
            },
            "8.03": {
                "title": "Physics III: Vibrations and Waves",
                "description": "Oscillations waves interference diffraction Fourier analysis",
                "level": "Undergraduate",
                "prerequisites": ["8.01"],
                "url": "https://ocw.mit.edu/courses/8-03sc-physics-iii-vibrations-and-waves-fall-2016/"
            },
            "8.04": {
                "title": "Quantum Physics I",
                "description": "Quantum mechanics wave function Schrodinger equation uncertainty principle",
                "level": "Undergraduate",
                "prerequisites": ["8.03"],
                "url": "https://ocw.mit.edu/courses/8-04-quantum-physics-i-spring-2016/"
            },
            "8.05": {
                "title": "Quantum Physics II",
                "description": "Advanced quantum mechanics operators angular momentum perturbation theory",
                "level": "Undergraduate",
                "prerequisites": ["8.04"],
                "url": "https://ocw.mit.edu/courses/8-05-quantum-physics-ii-fall-2013/"
            },
            "8.06": {
                "title": "Quantum Physics III",
                "description": "Scattering theory relativistic quantum mechanics many-body systems",
                "level": "Undergraduate",
                "prerequisites": ["8.05"],
                "url": "https://ocw.mit.edu/courses/8-06-quantum-physics-iii-spring-2018/"
            },
            "8.033": {
                "title": "Relativity",
                "description": "Special general relativity Lorentz transformations spacetime",
                "level": "Undergraduate",
                "prerequisites": ["8.01", "8.02"],
                "url": "https://ocw.mit.edu/courses/8-033-relativity-fall-2006/"
            },
            "8.044": {
                "title": "Statistical Physics I",
                "description": "Thermodynamics statistical mechanics entropy partition functions",
                "level": "Undergraduate",
                "prerequisites": ["8.01"],
                "url": "https://ocw.mit.edu/courses/8-044-statistical-physics-i-spring-2013/"
            },
            "18.01": {
                "title": "Single Variable Calculus",
                "description": "Derivatives integrals limits series fundamental theorem calculus",
                "level": "Undergraduate",
                "prerequisites": [],
                "url": "https://ocw.mit.edu/courses/18-01sc-single-variable-calculus-fall-2010/"
            },
            "18.02": {
                "title": "Multivariable Calculus",
                "description": "Partial derivatives multiple integrals vector calculus gradient divergence curl",
                "level": "Undergraduate",
                "prerequisites": ["18.01"],
                "url": "https://ocw.mit.edu/courses/18-02sc-multivariable-calculus-fall-2010/"
            },
            "18.03": {
                "title": "Differential Equations",
                "description": "Ordinary differential equations linear systems Laplace transforms",
                "level": "Undergraduate",
                "prerequisites": ["18.02"],
                "url": "https://ocw.mit.edu/courses/18-03sc-differential-equations-fall-2011/"
            },
            "18.06": {
                "title": "Linear Algebra",
                "description": "Matrices vectors eigenvalues eigenvectors linear transformations",
                "level": "Undergraduate",
                "prerequisites": ["18.02"],
                "url": "https://ocw.mit.edu/courses/18-06sc-linear-algebra-fall-2011/"
            }
        }

    def _prepare_course_vectors(self):
        '''Prepare TF-IDF vectors for courses'''
        course_texts = []
        self.course_order = []

        for course_num, info in self.courses.items():
            course_text = f"{info['title']} {info['description']}"
            course_texts.append(course_text)
            self.course_order.append(course_num)

        self.course_vectors = self.vectorizer.fit_transform(course_texts)

    def map_concepts_to_courses(self, concepts: List[Dict]) -> List[Dict]:
        '''Map concepts to courses using TF-IDF similarity'''
        if not concepts:
            return []

        # Create text from concepts
        concept_text = ' '.join([
            f"{c['concept']} {' '.join(c.get('matched_keywords', []))}"
            for c in concepts[:10]
        ])

        if not concept_text:
            return []

        # Vectorize concept text
        try:
            concept_vector = self.vectorizer.transform([concept_text])
        except:
            # If vectorizer fails, return default courses
            return self._get_default_courses()

        # Calculate similarities
        similarities = cosine_similarity(concept_vector, self.course_vectors)[0]

        # Get top courses
        top_indices = np.argsort(similarities)[::-1][:10]

        recommendations = []
        for idx in top_indices:
            if similarities[idx] > 0:
                course_num = self.course_order[idx]
                course_info = self.courses[course_num]

                recommendations.append({
                    'course_number': course_num,
                    'title': course_info['title'],
                    'description': course_info['description'][:200],
                    'url': course_info['url'],
                    'level': course_info['level'],
                    'similarity_score': float(similarities[idx]),
                    'prerequisites': course_info['prerequisites']
                })

        return recommendations

    def _get_default_courses(self):
        '''Return default introductory courses'''
        defaults = ['8.01', '8.02', '18.01', '18.02']
        return [
            {
                'course_number': num,
                'title': self.courses[num]['title'],
                'description': self.courses[num]['description'][:200],
                'url': self.courses[num]['url'],
                'level': self.courses[num]['level'],
                'similarity_score': 0.5,
                'prerequisites': self.courses[num]['prerequisites']
            }
            for num in defaults if num in self.courses
        ]

    def generate_learning_path(self, recommended_courses: List[Dict]) -> List[Dict]:
        '''Generate learning path with prerequisites'''
        if not recommended_courses:
            return []

        # Get all required courses including prerequisites
        all_courses = set()
        for course in recommended_courses[:5]:
            course_num = course['course_number']
            all_courses.add(course_num)
            if course_num in self.courses:
                all_courses.update(self.courses[course_num]['prerequisites'])

        # Simple ordering by course number (usually reflects difficulty)
        path = []

        # Math courses first
        math_courses = sorted([c for c in all_courses if c.startswith('18.')])
        physics_courses = sorted([c for c in all_courses if c.startswith('8.')])

        for course_num in math_courses + physics_courses:
            if course_num in self.courses:
                path.append({
                    'course_number': course_num,
                    'title': self.courses[course_num]['title'],
                    'level': self.courses[course_num]['level']
                })

        return path