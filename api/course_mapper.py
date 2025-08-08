import numpy as np
from typing import List, Dict, Tuple
from sklearn.metrics.pairwise import cosine_similarity
import json
import faiss
from .bert_models import SciBERTProcessor
import logging

logger = logging.getLogger(__name__)


class EmbeddingCourseMapper:
    """Map concepts to courses using embeddings"""

    def __init__(self):
        self.scibert = SciBERTProcessor()
        self.courses = self._initialize_course_database()
        self.course_index = self._build_course_index()

    def _initialize_course_database(self) -> Dict:
        """Initialize MIT courses with detailed descriptions"""
        courses = {
            "8.01": {
                "title": "Physics I: Classical Mechanics",
                "description": "Newtonian mechanics momentum energy conservation laws kinematics dynamics rotational motion oscillations",
                "topics": ["Newton's laws", "Conservation of energy", "Momentum", "Rotational dynamics",
                           "Harmonic oscillator"],
                "level": "Undergraduate",
                "prerequisites": ["18.01"],
                "url": "https://ocw.mit.edu/courses/8-01sc-classical-mechanics-fall-2016/"
            },
            "8.02": {
                "title": "Physics II: Electricity and Magnetism",
                "description": "Electrostatics magnetostatics Maxwell equations electromagnetic fields waves Gauss law Ampere law Faraday",
                "topics": ["Coulomb's law", "Gauss's law", "Capacitance", "Magnetic fields", "Faraday's law",
                           "Maxwell's equations"],
                "level": "Undergraduate",
                "prerequisites": ["8.01", "18.02"],
                "url": "https://ocw.mit.edu/courses/8-02-physics-ii-electricity-and-magnetism-spring-2007/"
            },
            "8.03": {
                "title": "Physics III: Vibrations and Waves",
                "description": "Oscillations waves mechanical electromagnetic wave equation dispersion interference diffraction Fourier",
                "topics": ["Simple harmonic motion", "Coupled oscillators", "Wave equation", "Electromagnetic waves",
                           "Interference"],
                "level": "Undergraduate",
                "prerequisites": ["8.01", "18.03"],
                "url": "https://ocw.mit.edu/courses/8-03sc-physics-iii-vibrations-and-waves-fall-2016/"
            },
            "8.04": {
                "title": "Quantum Physics I",
                "description": "Introduction quantum mechanics wave function Schrodinger equation uncertainty principle quantum states",
                "topics": ["Wave-particle duality", "Schrodinger equation", "Quantum wells", "Harmonic oscillator",
                           "Hydrogen atom"],
                "level": "Undergraduate",
                "prerequisites": ["8.03", "18.03"],
                "url": "https://ocw.mit.edu/courses/8-04-quantum-physics-i-spring-2016/"
            },
            "8.05": {
                "title": "Quantum Physics II",
                "description": "Advanced quantum mechanics operators eigenvalues angular momentum spin perturbation theory",
                "topics": ["Quantum operators", "Angular momentum", "Spin", "Perturbation theory", "Scattering"],
                "level": "Undergraduate",
                "prerequisites": ["8.04"],
                "url": "https://ocw.mit.edu/courses/8-05-quantum-physics-ii-fall-2013/"
            },
            "8.06": {
                "title": "Quantum Physics III",
                "description": "Advanced quantum mechanics scattering theory relativistic quantum mechanics field theory introduction",
                "topics": ["Time-dependent perturbation", "Scattering theory", "Relativistic quantum mechanics",
                           "Many-body systems"],
                "level": "Undergraduate",
                "prerequisites": ["8.05"],
                "url": "https://ocw.mit.edu/courses/8-06-quantum-physics-iii-spring-2018/"
            },
            "8.223": {
                "title": "Classical Mechanics II",
                "description": "Lagrangian Hamiltonian mechanics variational principles canonical transformations Hamilton-Jacobi theory",
                "topics": ["Lagrangian formulation", "Hamiltonian formulation", "Canonical transformations",
                           "Hamilton-Jacobi theory"],
                "level": "Undergraduate",
                "prerequisites": ["8.01", "18.03"],
                "url": "https://ocw.mit.edu/courses/8-223-classical-mechanics-ii-january-iap-2017/"
            },
            "8.033": {
                "title": "Relativity",
                "description": "Special general relativity Lorentz transformations spacetime metrics Einstein field equations",
                "topics": ["Special relativity", "Lorentz transformations", "General relativity basics",
                           "Spacetime geometry"],
                "level": "Undergraduate",
                "prerequisites": ["8.01", "8.02", "18.03"],
                "url": "https://ocw.mit.edu/courses/8-033-relativity-fall-2006/"
            },
            "8.044": {
                "title": "Statistical Physics I",
                "description": "Thermodynamics statistical mechanics entropy partition functions ensembles phase transitions",
                "topics": ["Thermodynamic laws", "Statistical ensembles", "Partition functions", "Ideal gases",
                           "Phase transitions"],
                "level": "Undergraduate",
                "prerequisites": ["8.01", "8.02"],
                "url": "https://ocw.mit.edu/courses/8-044-statistical-physics-i-spring-2013/"
            },
            "8.321": {
                "title": "Quantum Theory I",
                "description": "Graduate quantum mechanics path integrals symmetries relativistic quantum mechanics field theory",
                "topics": ["Path integral formulation", "Symmetries in quantum mechanics",
                           "Relativistic quantum mechanics"],
                "level": "Graduate",
                "prerequisites": ["8.06"],
                "url": "https://ocw.mit.edu/courses/8-321-quantum-theory-i-fall-2017/"
            },
            "8.333": {
                "title": "Statistical Mechanics I",
                "description": "Graduate statistical mechanics phase transitions critical phenomena renormalization group theory",
                "topics": ["Phase transitions", "Critical phenomena", "Renormalization group", "Monte Carlo methods"],
                "level": "Graduate",
                "prerequisites": ["8.044"],
                "url": "https://ocw.mit.edu/courses/8-333-statistical-mechanics-i-statistical-mechanics-of-particles-fall-2013/"
            },
            "8.962": {
                "title": "General Relativity",
                "description": "Graduate general relativity differential geometry tensor calculus Einstein equations black holes cosmology",
                "topics": ["Tensor calculus", "Einstein field equations", "Schwarzschild solution", "Black holes",
                           "Cosmology"],
                "level": "Graduate",
                "prerequisites": ["8.033", "18.06"],
                "url": "https://ocw.mit.edu/courses/8-962-general-relativity-spring-2020/"
            },
            "18.01": {
                "title": "Single Variable Calculus",
                "description": "Differentiation integration limits continuity derivatives integrals fundamental theorem calculus",
                "topics": ["Limits", "Derivatives", "Integrals", "Series", "Applications"],
                "level": "Undergraduate",
                "prerequisites": [],
                "url": "https://ocw.mit.edu/courses/18-01sc-single-variable-calculus-fall-2010/"
            },
            "18.02": {
                "title": "Multivariable Calculus",
                "description": "Vector calculus partial derivatives multiple integrals gradient divergence curl Stokes Green theorems",
                "topics": ["Partial derivatives", "Multiple integrals", "Vector fields", "Line integrals",
                           "Surface integrals"],
                "level": "Undergraduate",
                "prerequisites": ["18.01"],
                "url": "https://ocw.mit.edu/courses/18-02sc-multivariable-calculus-fall-2010/"
            },
            "18.03": {
                "title": "Differential Equations",
                "description": "Ordinary differential equations linear systems Laplace transforms Fourier series boundary value problems",
                "topics": ["First-order ODEs", "Second-order ODEs", "Systems of ODEs", "Laplace transforms",
                           "Fourier series"],
                "level": "Undergraduate",
                "prerequisites": ["18.02"],
                "url": "https://ocw.mit.edu/courses/18-03sc-differential-equations-fall-2011/"
            },
            "18.06": {
                "title": "Linear Algebra",
                "description": "Matrices vectors eigenvalues eigenvectors determinants linear transformations vector spaces",
                "topics": ["Matrix operations", "Eigenvalues", "Vector spaces", "Linear transformations",
                           "Applications"],
                "level": "Undergraduate",
                "prerequisites": ["18.02"],
                "url": "https://ocw.mit.edu/courses/18-06sc-linear-algebra-fall-2011/"
            }
        }

        # Generate embeddings for each course
        for course_num, course_info in courses.items():
            # Combine title and description for embedding
            course_text = f"{course_info['title']} {course_info['description']}"
            embedding = self.scibert.get_embeddings([course_text])[0]
            course_info['embedding'] = embedding

        return courses

    def _build_course_index(self) -> faiss.Index:
        """Build FAISS index for course embeddings"""
        embeddings = []
        self.course_numbers = []

        for num, info in self.courses.items():
            embeddings.append(info['embedding'])
            self.course_numbers.append(num)

        embeddings = np.array(embeddings).astype('float32')

        # Create FAISS index
        dimension = embeddings.shape[1]
        index = faiss.IndexFlatIP(dimension)

        # Normalize for cosine similarity
        faiss.normalize_L2(embeddings)
        index.add(embeddings)

        return index

    def map_concepts_to_courses(self, concepts: List[Dict], equations: List[Dict]) -> List[Dict]:
        """Map extracted concepts to relevant courses using embeddings"""

        # Create a combined representation of the document's concepts
        concept_texts = []
        weights = []

        for concept in concepts[:10]:  # Top 10 concepts
            concept_info = concept.get('concept', '')
            score = concept.get('score', 1.0)
            concept_texts.append(concept_info)
            weights.append(score)

        if not concept_texts:
            return []

        # Get embeddings for concepts
        concept_embeddings = self.scibert.get_embeddings(concept_texts)

        # Weighted average of concept embeddings
        weights = np.array(weights) / np.sum(weights)
        weighted_embedding = np.average(concept_embeddings, axis=0, weights=weights)
        weighted_embedding = weighted_embedding.astype('float32').reshape(1, -1)
        faiss.normalize_L2(weighted_embedding)

        # Search for similar courses
        k = min(10, len(self.courses))
        distances, indices = self.course_index.search(weighted_embedding, k)

        # Build recommendations
        recommendations = []
        for dist, idx in zip(distances[0], indices[0]):
            course_num = self.course_numbers[idx]
            course_info = self.courses[course_num]

            # Calculate additional relevance factors
            complexity_match = self._calculate_complexity_match(concepts, course_info)

            recommendations.append({
                'course_number': course_num,
                'title': course_info['title'],
                'description': course_info['description'][:200] + '...',
                'url': course_info['url'],
                'level': course_info['level'],
                'similarity_score': float(dist),
                'complexity_match': complexity_match,
                'overall_score': float(dist) * 0.7 + complexity_match * 0.3,
                'topics': course_info['topics'][:5],
                'prerequisites': course_info['prerequisites']
            })

        # Sort by overall score
        recommendations.sort(key=lambda x: x['overall_score'], reverse=True)

        return recommendations

    def _calculate_complexity_match(self, concepts: List[Dict], course_info: Dict) -> float:
        """Calculate how well the course complexity matches the content"""
        # Average concept level
        if not concepts:
            return 0.5

        avg_level = np.mean([c.get('level', 3) for c in concepts[:5]])

        # Map course level to numeric
        level_map = {
            'Undergraduate': 2,
            'Graduate': 4
        }
        course_level = level_map.get(course_info['level'], 3)

        # Calculate match (closer levels = higher score)
        diff = abs(avg_level - course_level)
        match_score = max(0, 1 - (diff / 3))

        return match_score

    def generate_learning_path(self, recommended_courses: List[Dict]) -> List[Dict]:
        """Generate an optimal learning path considering prerequisites"""
        # Extract course numbers
        course_nums = [c['course_number'] for c in recommended_courses[:8]]

        # Build dependency graph
        dependencies = {}
        all_courses = set(course_nums)

        for num in course_nums:
            if num in self.courses:
                prereqs = self.courses[num]['prerequisites']
                dependencies[num] = prereqs
                all_courses.update(prereqs)

        # Topological sort for learning path
        visited = set()
        path = []

        def visit(course):
            if course in visited:
                return
            visited.add(course)
            if course in dependencies:
                for prereq in dependencies[course]:
                    if prereq in all_courses:
                        visit(prereq)
            path.append(course)

        for course in all_courses:
            visit(course)

        # Build detailed path with course info
        learning_path = []
        for course_num in path:
            if course_num in self.courses:
                course_info = self.courses[course_num]
                learning_path.append({
                    'course_number': course_num,
                    'title': course_info['title'],
                    'level': course_info['level'],
                    'url': course_info['url']
                })

        return learning_path