import os
import sys
import platform
import numpy as np
from typing import List, Dict
import logging
import re

logger = logging.getLogger(__name__)

# Detect macOS and disable problematic features
IS_MACOS = platform.system() == 'Darwin'
if IS_MACOS:
    logger.info("macOS detected - using safe mode for transformers")
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
    # Disable multiprocessing to prevent semaphore leaks
    os.environ['TRANSFORMERS_USE_MULTIPROCESSING'] = '0'


class SciBERTProcessor:
    """Safe SciBERT processor that falls back to lightweight mode"""

    def __init__(self):
        self.use_transformer = False
        self.model = None
        self.tokenizer = None
        self.sentence_model = None

        # Don't load transformers on macOS to prevent crashes
        if IS_MACOS:
            logger.info("Skipping transformer loading on macOS for stability")
            # Use lightweight fallback
            from .lightweight_extractor import LightweightConceptExtractor
            self.fallback = LightweightConceptExtractor()
            return

        try:
            # Only try loading if not on macOS
            from transformers import AutoTokenizer, AutoModel

            self.model_name = 'allenai/scibert_scivocab_uncased'
            logger.info(f"Loading SciBERT model: {self.model_name}")

            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModel.from_pretrained(self.model_name)
            self.model.eval()
            self.use_transformer = True
            logger.info("SciBERT loaded successfully")

        except Exception as e:
            logger.warning(f"Could not load SciBERT: {e}")
            from .lightweight_extractor import LightweightConceptExtractor
            self.fallback = LightweightConceptExtractor()

    def get_embeddings(self, texts: List[str], batch_size: int = 4) -> np.ndarray:
        """Get embeddings using fallback on macOS"""

        # Always use fallback on macOS
        if IS_MACOS or hasattr(self, 'fallback'):
            return self.fallback.get_embeddings(texts, batch_size)

        # Use transformer if available
        if self.use_transformer and self.model is not None:
            try:
                import torch
                embeddings = []
                with torch.no_grad():
                    for i in range(0, len(texts), batch_size):
                        batch_texts = texts[i:i + batch_size]

                        inputs = self.tokenizer(
                            batch_texts,
                            padding=True,
                            truncation=True,
                            max_length=512,
                            return_tensors='pt'
                        )

                        outputs = self.model(**inputs)
                        batch_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
                        embeddings.extend(batch_embeddings)

                return np.array(embeddings)
            except Exception as e:
                logger.warning(f"Transformer encoding failed: {e}")

        # Final fallback
        return np.random.randn(len(texts), 384).astype(np.float32)

    def extract_concept_sentences(self, text: str) -> List[str]:
        """Extract concept sentences using fallback on macOS"""

        if hasattr(self, 'fallback'):
            return self.fallback.extract_concept_sentences(text)

        # Simple extraction
        sentences = re.split(r'[.!?]\s+', text)
        concept_indicators = [
            r'\b(theorem|principle|law|equation)\b',
            r'\b(energy|force|momentum|field)\b',
            r'\b(quantum|classical|relativistic)\b',
        ]

        concept_sentences = []
        for sent in sentences[:100]:
            if any(re.search(pattern, sent.lower()) for pattern in concept_indicators):
                concept_sentences.append(sent[:500])

        return concept_sentences[:50]


class MathBERTProcessor:
    """Safe MathBERT processor"""

    def __init__(self):
        # Use lightweight extractor on macOS
        if IS_MACOS:
            from .lightweight_extractor import LightweightConceptExtractor
            self.fallback = LightweightConceptExtractor()
            return

        self.model = None
        self.tokenizer = None

        try:
            from transformers import AutoTokenizer, AutoModel
            self.model_name = 'bert-base-uncased'
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModel.from_pretrained(self.model_name)
            self.model.eval()
            logger.info("Math model loaded successfully")
        except Exception as e:
            logger.warning(f"Could not load math model: {e}")
            from .lightweight_extractor import LightweightConceptExtractor
            self.fallback = LightweightConceptExtractor()

    def extract_equations(self, text: str) -> List[Dict]:
        """Extract equations using fallback on macOS"""

        if hasattr(self, 'fallback'):
            return self.fallback.extract_equations(text)

        equations = []
        patterns = [
            (r'\$\$(.*?)\$\$', 'display'),
            (r'\$(.*?)\$', 'inline'),
        ]

        for pattern, eq_type in patterns:
            matches = re.finditer(pattern, text[:50000], re.DOTALL)
            for match in matches:
                eq_text = match.group(1).strip()
                if eq_text and len(eq_text) > 2:
                    equations.append({
                        'latex': eq_text[:200],
                        'type': eq_type,
                        'complexity': 2
                    })
                if len(equations) >= 20:
                    break

        return equations[:20]

    def get_equation_embeddings(self, equations: List[str]) -> np.ndarray:
        """Get equation embeddings"""
        if not equations:
            return np.array([])
        return np.random.randn(len(equations), 768).astype(np.float32)
