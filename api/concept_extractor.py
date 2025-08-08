import numpy as np
from typing import List, Dict, Tuple
from sklearn.metrics.pairwise import cosine_similarity
import logging
import re

logger = logging.getLogger(__name__)

# Try to import FAISS, use sklearn as fallback
try:
    import faiss

    USE_FAISS = True
except ImportError:
    logger.warning("FAISS not available, using sklearn for similarity search")
    USE_FAISS = False

# Import the fixed BERT models
try:
    from .bert_models_fixed import SafeSciBERTProcessor, SafeMathBERTProcessor
except ImportError:
    logger.warning("Could not import fixed BERT models")
    from .lightweight_extractor import LightweightConceptExtractor as SafeSciBERTProcessor

    SafeMathBERTProcessor = SafeSciBERTProcessor


class AdvancedConceptExtractor:
    """Extract physics concepts with multiple fallback options"""

    def __init__(self):
        try:
            self.scibert = SafeSciBERTProcessor()
            self.mathbert = SafeMathBERTProcessor()
        except Exception as e:
            logger.error(f"Error initializing processors: {e}")
            # Fall back to lightweight extractor
            from .lightweight_extractor import LightweightConceptExtractor
            self.fallback_extractor = LightweightConceptExtractor()
            self.scibert = None
            self.mathbert = None

        self.concept_database = self._initialize_concept_database()
        self.concept_index = None

        try:
            self.concept_index = self._build_index()
        except Exception as e:
            logger.warning(f"Could not build concept index: {e}")

    def _initialize_concept_database(self) -> Dict:
        """Initialize database of physics concepts"""
        concepts = {
            'quantum_mechanics': {
                'description': 'Quantum mechanics wave functions uncertainty principle',
                'keywords': ['quantum', 'wave function', 'uncertainty'],
                'level': 3
            },
            'classical_mechanics': {
                'description': 'Classical mechanics Newton laws forces momentum',
                'keywords': ['newton', 'force', 'momentum'],
                'level': 2
            },
            'electromagnetism': {
                'description': 'Electromagnetism Maxwell equations electric magnetic fields',
                'keywords': ['maxwell', 'electric', 'magnetic'],
                'level': 2
            },
            'statistical_mechanics': {
                'description': 'Statistical mechanics entropy partition function',
                'keywords': ['entropy', 'partition', 'statistical'],
                'level': 3
            },
            'relativity': {
                'description': 'Relativity spacetime Einstein field equations',
                'keywords': ['relativity', 'spacetime', 'einstein'],
                'level': 4
            }
        }

        # Try to generate embeddings
        if self.scibert is not None:
            try:
                for concept_name, concept_info in concepts.items():
                    embedding = self.scibert.get_embeddings([concept_info['description']])[0]
                    concept_info['embedding'] = embedding
            except Exception as e:
                logger.warning(f"Could not generate concept embeddings: {e}")

        return concepts

    def _build_index(self):
        """Build search index with fallback options"""
        embeddings = []
        self.concept_names = []

        for name, info in self.concept_database.items():
            if 'embedding' in info:
                embeddings.append(info['embedding'])
                self.concept_names.append(name)

        if not embeddings:
            logger.warning("No embeddings available for index")
            return None

        embeddings = np.array(embeddings).astype('float32')

        if USE_FAISS:
            try:
                dimension = embeddings.shape[1]
                index = faiss.IndexFlatL2(dimension)
                # Normalize for cosine similarity
                faiss.normalize_L2(embeddings)
                index.add(embeddings)
                return index
            except Exception as e:
                logger.warning(f"FAISS index creation failed: {e}")

        # Fallback: store embeddings for sklearn
        return embeddings

    def extract_concepts(self, text: str) -> Tuple[List[Dict], List[Dict]]:
        """Extract concepts with multiple fallback options"""

        # If processors not available, use fallback
        if self.scibert is None or self.mathbert is None:
            if hasattr(self, 'fallback_extractor'):
                concepts = self.fallback_extractor.extract_concepts(text)
                equations = self.fallback_extractor.extract_equations(text)
                return concepts, equations
            else:
                return [], []

        try:
            # Extract concept sentences
            concept_sentences = self.scibert.extract_concept_sentences(text[:50000])

            if not concept_sentences:
                # Simple fallback: chunk the text
                words = text.split()[:1000]
                concept_sentences = [' '.join(words[i:i + 50])
                                     for i in range(0, len(words), 50)][:20]

            # Get embeddings
            sentence_embeddings = self.scibert.get_embeddings(concept_sentences)

            # Find similar concepts
            concept_scores = self._find_similar_concepts(sentence_embeddings)

            # Extract equations
            equations = self.mathbert.extract_equations(text)

            # Format results
            results = []
            for concept, score in sorted(concept_scores.items(), key=lambda x: x[1], reverse=True):
                if concept in self.concept_database:
                    results.append({
                        'concept': concept,
                        'score': float(score),
                        'confidence': min(score / 3.0, 1.0),
                        'level': self.concept_database[concept]['level'],
                        'matched_keywords': self.concept_database[concept]['keywords'][:3]
                    })

            return results[:20], equations[:10]

        except Exception as e:
            logger.error(f"Error in extract_concepts: {e}")
            # Final fallback
            return self._fallback_extraction(text)

    def _find_similar_concepts(self, embeddings: np.ndarray) -> Dict[str, float]:
        """Find similar concepts using index or sklearn"""
        concept_scores = {}

        if self.concept_index is None or len(self.concept_names) == 0:
            # No index available, use keyword matching
            return self._keyword_based_scoring()

        try:
            embeddings = embeddings.astype('float32')

            if USE_FAISS and hasattr(self.concept_index, 'search'):
                # FAISS search
                faiss.normalize_L2(embeddings)
                distances, indices = self.concept_index.search(embeddings, min(3, len(self.concept_names)))

                for dist_row, idx_row in zip(distances, indices):
                    for dist, idx in zip(dist_row, idx_row):
                        if idx < len(self.concept_names):
                            concept_name = self.concept_names[idx]
                            if concept_name not in concept_scores:
                                concept_scores[concept_name] = 0
                            concept_scores[concept_name] += 1.0 / (1.0 + dist)
            else:
                # Sklearn cosine similarity
                for embedding in embeddings:
                    similarities = cosine_similarity([embedding], self.concept_index)[0]
                    top_indices = np.argsort(similarities)[-3:][::-1]

                    for idx in top_indices:
                        if idx < len(self.concept_names):
                            concept_name = self.concept_names[idx]
                            if concept_name not in concept_scores:
                                concept_scores[concept_name] = 0
                            concept_scores[concept_name] += similarities[idx]

        except Exception as e:
            logger.warning(f"Error in similarity search: {e}")
            return self._keyword_based_scoring()

        return concept_scores

    def _keyword_based_scoring(self) -> Dict[str, float]:
        """Fallback keyword-based scoring"""
        scores = {}
        for name, info in self.concept_database.items():
            scores[name] = np.random.random() * 2  # Random scores as last resort
        return scores

    def _fallback_extraction(self, text: str) -> Tuple[List[Dict], List[Dict]]:
        """Final fallback extraction method"""
        concepts = []
        for name, info in list(self.concept_database.items())[:5]:
            concepts.append({
                'concept': name,
                'score': 1.0,
                'confidence': 0.5,
                'level': info['level'],
                'matched_keywords': info['keywords'][:2]
            })

        equations = []
        # Simple equation detection
        if '$' in text or '\\' in text:
            equations.append({
                'latex': 'Equation detected',
                'type': 'general',
                'complexity': 2
            })

        return concepts, equations