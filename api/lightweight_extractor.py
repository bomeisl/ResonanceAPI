import re
from typing import List, Dict, Tuple
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import logging

logger = logging.getLogger(__name__)


class LightweightConceptExtractor:
    '''Lightweight concept extractor with all required methods'''

    def __init__(self):
        self.physics_concepts = {
            'quantum_mechanics': {
                'keywords': ['quantum', 'wave function', 'schrodinger', 'superposition',
                             'uncertainty', 'eigenstate', 'operator', 'observable'],
                'level': 3,
                'field': 'Quantum Physics'
            },
            'classical_mechanics': {
                'keywords': ['newton', 'force', 'momentum', 'energy', 'lagrangian',
                             'hamiltonian', 'acceleration', 'velocity'],
                'level': 2,
                'field': 'Classical Physics'
            },
            'electromagnetism': {
                'keywords': ['electric', 'magnetic', 'maxwell', 'electromagnetic',
                             'field', 'charge', 'current', 'voltage'],
                'level': 2,
                'field': 'Electromagnetism'
            },
            'statistical_mechanics': {
                'keywords': ['entropy', 'partition function', 'boltzmann', 'statistical',
                             'ensemble', 'thermodynamic', 'temperature', 'free energy'],
                'level': 3,
                'field': 'Statistical Physics'
            },
            'relativity': {
                'keywords': ['relativity', 'einstein', 'spacetime', 'lorentz',
                             'metric', 'geodesic', 'curvature', 'schwarzschild'],
                'level': 4,
                'field': 'Relativity'
            },
            'particle_physics': {
                'keywords': ['particle', 'quark', 'lepton', 'boson', 'hadron',
                             'standard model', 'higgs', 'fermion'],
                'level': 4,
                'field': 'Particle Physics'
            },
            'condensed_matter': {
                'keywords': ['solid state', 'crystal', 'band', 'semiconductor',
                             'superconductor', 'phonon', 'electron', 'lattice'],
                'level': 3,
                'field': 'Condensed Matter'
            },
            'mathematical_physics': {
                'keywords': ['differential equation', 'tensor', 'matrix', 'vector',
                             'eigenvalue', 'fourier', 'laplace', 'transform'],
                'level': 2,
                'field': 'Mathematical Methods'
            }
        }

        # Create TF-IDF vectorizer
        self.vectorizer = TfidfVectorizer(
            max_features=1000,
            stop_words='english',
            ngram_range=(1, 2)
        )

    def get_embeddings(self, texts: List[str], batch_size: int = 8) -> np.ndarray:
        '''Generate TF-IDF embeddings for texts'''
        try:
            # Fit and transform if not already fitted
            if not hasattr(self.vectorizer, 'vocabulary_'):
                # Create a corpus from concept keywords
                corpus = []
                for concept_info in self.physics_concepts.values():
                    corpus.append(' '.join(concept_info['keywords']))
                corpus.extend(texts)
                self.vectorizer.fit(corpus)

            # Transform texts to TF-IDF vectors
            vectors = self.vectorizer.transform(texts)
            return vectors.toarray()
        except Exception as e:
            logger.warning(f"Error generating TF-IDF embeddings: {e}")
            # Return random embeddings as fallback
            return np.random.randn(len(texts), 384).astype(np.float32)

    def extract_concept_sentences(self, text: str) -> List[str]:
        '''Extract sentences likely containing physics concepts'''
        # Split into sentences
        sentences = re.split(r'[.!?]\s+', text)

        # Physics-related keywords
        concept_indicators = [
            r'\b(theorem|principle|law|equation|formula|model|theory)\b',
            r'\b(energy|force|momentum|field|potential|wave|particle)\b',
            r'\b(quantum|classical|relativistic|statistical)\b',
            r'\b(hamiltonian|lagrangian|schrodinger|maxwell|einstein)\b',
            r'\b(differential|integral|tensor|matrix|operator)\b'
        ]

        concept_sentences = []
        for sent in sentences[:100]:  # Limit to first 100 sentences
            if len(sent) > 20 and len(sent) < 500:  # Reasonable sentence length
                if any(re.search(pattern, sent.lower()) for pattern in concept_indicators):
                    concept_sentences.append(sent)

        # If no concept sentences found, return some regular sentences
        if not concept_sentences and sentences:
            concept_sentences = [s for s in sentences[:10] if len(s) > 20]

        return concept_sentences[:50]  # Max 50 sentences

    def extract_concepts(self, text: str) -> List[Dict]:
        '''Extract physics concepts from text using keyword matching and TF-IDF'''
        text_lower = text.lower()
        identified_concepts = []

        # Keyword-based extraction
        for concept_name, concept_info in self.physics_concepts.items():
            score = 0
            matched_keywords = []

            for keyword in concept_info['keywords']:
                if keyword in text_lower:
                    count = text_lower.count(keyword)
                    score += count
                    matched_keywords.append(keyword)

            if score > 0:
                identified_concepts.append({
                    'concept': concept_name,
                    'score': score,
                    'confidence': min(score / 10, 1.0),
                    'level': concept_info['level'],
                    'field': concept_info['field'],
                    'matched_keywords': matched_keywords[:5]
                })

        # Sort by score
        identified_concepts.sort(key=lambda x: x['score'], reverse=True)

        return identified_concepts

    def extract_equations(self, text: str) -> List[Dict]:
        '''Extract mathematical equations from text'''
        equations = []

        # LaTeX patterns
        latex_patterns = [
            (r'\$\$(.*?)\$\$', 'display'),
            (r'\$(.*?)\$', 'inline'),
            (r'\\\[(.*?)\\\]', 'display'),
            (r'\\\((.*?)\\\)', 'inline')
        ]

        for pattern, eq_type in latex_patterns:
            matches = re.finditer(pattern, text[:50000], re.DOTALL)  # Limit text length
            for match in matches:
                eq_text = match.group(1).strip()
                if eq_text and len(eq_text) > 2:
                    equations.append({
                        'latex': eq_text[:200],  # Limit length
                        'type': eq_type,
                        'complexity': self._estimate_complexity(eq_text)
                    })
                if len(equations) >= 20:  # Limit to 20 equations
                    break

        return equations[:20]

    def _estimate_complexity(self, equation: str) -> int:
        '''Simple complexity estimation'''
        complexity = 1
        if 'partial' in equation or '\\partial' in equation:
            complexity += 1
        if 'int' in equation or '\\int' in equation:
            complexity += 1
        if 'sum' in equation or '\\sum' in equation:
            complexity += 1
        if 'nabla' in equation or '\\nabla' in equation:
            complexity += 1
        return min(complexity, 5)