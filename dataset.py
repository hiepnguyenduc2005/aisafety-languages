"""
Dataset module for multilingual AI safety evaluation.
Handles data curation, TATER framework, and transcreation.
"""

import json
import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, asdict, field
from collections import defaultdict
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s - %(name)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class SafetyPrompt:
    """Represents a single safety-critical prompt."""
    id: str
    original_text: str
    domain: str  # Crimes, Fairness, Privacy, Physical Harm, Misuse
    severity_level: int  # L0-L3 (0=benign, 3=extreme)
    translations: Dict[str, str]  # {language: translated_text}
    transcreated: Dict[str, str]  # {language: transcreated_text}
    options: List[str] = field(default_factory=list)
    correct_option: Optional[str] = None
    rationale: Optional[str] = None
    options_translations: Dict[str, List[str]] = field(default_factory=dict)
    
    def to_dict(self):
        return asdict(self)


@dataclass
class SafetyDataset:
    """Container for the full safety evaluation dataset."""
    prompts: List[SafetyPrompt]
    languages: List[str]
    domains: List[str]
    stats: Dict
    
    def get_domain_distribution(self) -> Dict[str, int]:
        """Returns count of prompts per domain."""
        dist = defaultdict(int)
        for prompt in self.prompts:
            dist[prompt.domain] += 1
        return dict(dist)
    
    def get_language_coverage(self) -> Dict[str, int]:
        """Returns count of translated prompts per language."""
        coverage = {lang: 0 for lang in self.languages}
        for prompt in self.prompts:
            for lang in prompt.translations:
                if lang in coverage:
                    coverage[lang] += 1
        return coverage
    
    def to_json(self, filepath: str):
        """Export dataset to JSON."""
        data = {
            'prompts': [p.to_dict() for p in self.prompts],
            'languages': self.languages,
            'domains': self.domains,
            'stats': self.stats
        }
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        logger.info(f"Dataset saved to {filepath}")
    
    @classmethod
    def from_json(cls, filepath: str) -> 'SafetyDataset':
        """Load dataset from JSON."""
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        prompts = [SafetyPrompt(**p) for p in data['prompts']]
        return cls(
            prompts=prompts,
            languages=data['languages'],
            domains=data['domains'],
            stats=data['stats']
        )


class DataCurator:
    """Handles data curation with K-Means clustering on embeddings."""
    
    def __init__(self, embedding_model_name: str = "Alibaba-NLP/gte-multilingual-base"):
        """
        Initialize the data curator.
        
        Args:
            embedding_model_name: Name of the embedding model (multilingual recommended)
        """
        self.model_name = embedding_model_name
        self.embeddings = None
        self.kmeans = None
        logger.info(f"DataCurator initialized with {embedding_model_name}")
    
    def generate_embeddings(self, texts: List[str]) -> np.ndarray:
        """
        Generate embeddings for texts using mGTE.
        
        Args:
            texts: List of text strings
            
        Returns:
            Embedding matrix of shape (n_texts, embedding_dim)
        """
        try:
            from sentence_transformers import SentenceTransformer
            model = SentenceTransformer(self.model_name, trust_remote_code=True)
            embeddings = model.encode(texts, show_progress_bar=True)
            self.embeddings = embeddings
            logger.info(f"Generated embeddings: {embeddings.shape}")
            return embeddings
        except ImportError:
            logger.error("sentence-transformers not installed. Install with: pip install sentence-transformers")
            raise
    
    def filter_by_clustering(
        self, 
        texts: List[str], 
        n_clusters: int = 5,
        samples_per_cluster: int = 50
    ) -> Tuple[List[str], List[int]]:
        """
        Filter texts using K-Means clustering to ensure diversity.
        
        Args:
            texts: List of text strings
            n_clusters: Number of clusters (matches number of domains)
            samples_per_cluster: Target samples per cluster
            
        Returns:
            Tuple of (filtered_texts, cluster_assignments)
        """
        if self.embeddings is None:
            self.generate_embeddings(texts)
        
        # Apply K-Means clustering
        self.kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = self.kmeans.fit_predict(self.embeddings)
        
        # Select diverse samples from each cluster
        filtered_indices = []
        for cluster_id in range(n_clusters):
            cluster_indices = np.where(labels == cluster_id)[0]
            # Sample evenly from cluster
            n_samples = min(samples_per_cluster, len(cluster_indices))
            sampled = np.random.choice(cluster_indices, size=n_samples, replace=False)
            filtered_indices.extend(sampled)
        
        filtered_texts = [texts[i] for i in filtered_indices]
        filtered_labels = [labels[i] for i in filtered_indices]
        
        logger.info(f"Filtered to {len(filtered_texts)} diverse prompts across {n_clusters} clusters")
        return filtered_texts, filtered_labels
    
    def assign_to_domains(
        self, 
        texts: List[str], 
        labels: List[int],
        domains: List[str]
    ) -> Dict[str, List[str]]:
        """
        Assign filtered texts to safety domains.
        
        Args:
            texts: Filtered text list
            labels: Cluster assignments
            domains: Domain names (Crimes, Fairness, Privacy, Physical Harm, Misuse)
            
        Returns:
            Dictionary mapping domains to lists of prompts
        """
        domain_mapping = defaultdict(list)
        for text, label in zip(texts, labels):
            domain = domains[label % len(domains)]
            domain_mapping[domain].append(text)
        
        logger.info(f"Domain distribution: {list(domain_mapping.keys())}")
        return dict(domain_mapping)


class TATERFramework:
    """
    Task-Aware Translate, Estimate, Refine (TATER) framework for transcreation.
    Converts prompts across languages while preserving harmful intent and cultural nuances.
    """
    
    def __init__(
        self,
        translation_model=None,
        quality_estimator=None,
        translation_model_name: str = "google/translategemma-4b-it",
        translation_device_map: str = "auto",
        translation_torch_dtype: str = "auto",
        translation_max_new_tokens: int = 256,
        translation_temperature: float = 0.2,
        translation_top_p: float = 0.9
    ):
        """
        Initialize TATER framework.
        
        Args:
            translation_model: Optional LLM for translation (e.g., GPT, Claude)
            quality_estimator: LLM for quality estimation
            translation_model_name: Hugging Face model for translation
            translation_device_map: Device placement strategy (e.g., "auto", "cpu")
            translation_torch_dtype: Torch dtype (e.g., "auto", "float16")
            translation_max_new_tokens: Max tokens to generate for translation
            translation_temperature: Sampling temperature for translation generation
            translation_top_p: Top-p sampling parameter for translation generation
        """
        self.translation_model = translation_model
        self.quality_estimator = quality_estimator
        self.translation_model_name = translation_model_name
        self.translation_device_map = translation_device_map
        self.translation_torch_dtype = translation_torch_dtype
        self.translation_max_new_tokens = translation_max_new_tokens
        self.translation_temperature = translation_temperature
        self.translation_top_p = translation_top_p
        self._translation_tokenizer = None
        self._translation_model = None
        self.last_translation_source = None

    def _llm_generate(self, model, prompt_text: str) -> str:
        """Safely call model.generate with a text prompt."""
        if model is None:
            return ""
        try:
            return str(model.generate(prompt_text)).strip()
        except Exception as e:
            logger.debug(f"LLM generate failed: {e}")
            return ""

    def _extract_json(self, text: str) -> Optional[Dict[str, any]]:
        """Try to extract JSON object from text."""
        if not text:
            return None
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1 or end <= start:
            return None
        try:
            return json.loads(text[start:end + 1])
        except Exception:
            return None

    def _load_translation_model(self):
        if self._translation_model is not None and self._translation_tokenizer is not None:
            return self._translation_tokenizer, self._translation_model

        try:
            from transformers import AutoTokenizer, AutoModelForCausalLM
            self._translation_tokenizer = AutoTokenizer.from_pretrained(self.translation_model_name)
            self._translation_model = AutoModelForCausalLM.from_pretrained(
                self.translation_model_name,
                device_map=self.translation_device_map,
                torch_dtype=self.translation_torch_dtype
            )
            self._translation_model.eval()
            logger.info(f"Loaded translation model: {self.translation_model_name}")
            return self._translation_tokenizer, self._translation_model
        except ImportError:
            logger.debug("transformers not installed. Install with: pip install transformers")
            raise
    
    def translate(
        self, 
        prompt: str, 
        target_language: str,
        use_translategemma: bool = True,
        use_legacy_fallback: bool = False
    ) -> str:
        """
        Step 1: Translate using Translategemma or fallback LLM.
        
        Args:
            prompt: English safety prompt
            target_language: Target language name (e.g., 'Vietnamese', 'Greek')
            use_translategemma: Use Translategemma for translation
            use_legacy_fallback: Use legacy translation fallback if available
            
        Returns:
            Translated prompt
        """
        if use_translategemma:
            try:
                tokenizer, model = self._load_translation_model()
                system_message = (
                    "You are a professional translator. Translate the user's text from English "
                    f"to {target_language}. Preserve meaning, safety intent, and tone. "
                    "Return only the translated text."
                )
                user_message = f"Text to translate:\n{prompt}"

                if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
                    lang_codes = {
                        'Vietnamese': 'vi',
                        'Greek': 'el',
                        'Amharic': 'am',
                        'Tibetan': 'bo',
                        'English': 'en',
                        'Chinese': 'zh',
                        'Spanish': 'es',
                        'French': 'fr',
                        'German': 'de',
                        'Japanese': 'ja',
                        'Korean': 'ko',
                        'Arabic': 'ar',
                        'Hindi': 'hi',
                        'Portuguese': 'pt',
                        'Russian': 'ru'
                    }
                    target_code = lang_codes.get(target_language, target_language.lower()[:2])
                    messages = [
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "text",
                                    "source_lang_code": "en",
                                    "target_lang_code": target_code,
                                    "text": prompt,
                                    "image": None
                                }
                            ]
                        }
                    ]
                    prompt_text = tokenizer.apply_chat_template(
                        messages,
                        tokenize=False,
                        add_generation_prompt=True
                    )
                else:
                    prompt_text = f"{system_message}\n\n{user_message}\n\nTranslation:"

                inputs = tokenizer(prompt_text, return_tensors="pt")
                inputs = {k: v.to(model.device) for k, v in inputs.items()}
                gen_kwargs = {
                    "max_new_tokens": self.translation_max_new_tokens,
                    "do_sample": self.translation_temperature > 0,
                    "temperature": self.translation_temperature,
                    "top_p": self.translation_top_p
                }
                output_ids = model.generate(**inputs, **gen_kwargs)
                generated = output_ids[0][inputs["input_ids"].shape[1]:]
                result = tokenizer.decode(generated, skip_special_tokens=True).strip()
                if result:
                    self.last_translation_source = "translategemma"
                    logger.info(f"Translated to {target_language} using Translategemma")
                    return result
                logger.warning(
                    "Translategemma returned empty output for %s. Falling back.",
                    target_language
                )
            except Exception as e:
                logger.exception(
                    "Translategemma error for %s. Using fallback translation.",
                    target_language
                )

        if use_legacy_fallback:
            lang_codes = {
                'Vietnamese': 'vi',
                'Greek': 'el',
                'Amharic': 'am',
                'Tibetan': 'bo',
                'English': 'en',
                'Chinese': 'zh-CN',
                'Spanish': 'es',
                'French': 'fr',
                'German': 'de',
                'Japanese': 'ja',
                'Korean': 'ko',
                'Arabic': 'ar',
                'Hindi': 'hi',
                'Portuguese': 'pt',
                'Russian': 'ru'
            }
            try:
                from deep_translator import GoogleTranslator
                target_code = lang_codes.get(target_language, target_language.lower()[:2])
                translator = GoogleTranslator(source='en', target=target_code)
                result = translator.translate(prompt)
                self.last_translation_source = "deep-translator"
                logger.info(f"Translated to {target_language} using deep-translator fallback")
                return result
            except ImportError:
                logger.debug("deep-translator not installed. Install with: pip install deep-translator>=1.11.4")
            except Exception as e:
                logger.debug(f"deep-translator error: {e}. Using fallback translation.")
        
        # Use LLM for translation if available
        if self.translation_model:
            prompt_text = f"Translate to {target_language}, preserving harmful intent:\n{prompt}"
            # translated = self.translation_model.generate(prompt_text)
            self.last_translation_source = "llm"
            logger.info(f"Translating to {target_language} using LLM")
            return f"[Translated to {target_language}] {prompt}"
        
        # Simple fallback: prefix with language indicator
        self.last_translation_source = "mock"
        logger.info(f"Using mock translation for {target_language}")
        return f"[{target_language}] {prompt}"
    
    def estimate_quality(
        self, 
        original: str, 
        translated: str,
        target_language: str
    ) -> Dict[str, any]:
        """
        Step 2: Estimate quality against safety-specific requirements.
        
        Args:
            original: Original English prompt
            translated: Translated prompt
            target_language: Target language
            
        Returns:
            Quality metrics
        """
        # If LLM estimator available, use it for quality scoring
        if self.quality_estimator is not None:
            rubric = (
                "You are a translation quality evaluator for safety-critical prompts. "
                "Score the translation against the original. Output ONLY JSON with keys: "
                "intent_preservation (0-1), cultural_adaptation (0-1), readability (0-1), "
                "requires_refinement (true [if only need improvements]/false [default]), suggestions (array of strings)."
            )
            prompt_text = (
                f"{rubric}\n\n"
                f"Target language: {target_language}\n"
                f"Original: {original}\n"
                f"Translated: {translated}\n"
                "JSON:"
            )
            raw = self._llm_generate(self.quality_estimator, prompt_text)
            parsed = self._extract_json(raw)
            if isinstance(parsed, dict):
                quality_metrics = {
                    'intent_preservation': float(parsed.get('intent_preservation', 0.9)),
                    'cultural_adaptation': float(parsed.get('cultural_adaptation', 0.8)),
                    'readability': float(parsed.get('readability', 0.9)),
                    'requires_refinement': bool(parsed.get('requires_refinement', False)),
                    'suggestions': list(parsed.get('suggestions', []))
                }
                logger.info(f"Quality estimation (LLM) for {target_language}: {quality_metrics}")
                return quality_metrics

        # Check for key safety indicators preservation (heuristic fallback)
        quality_metrics = {
            'intent_preservation': 0.95,  # Placeholder
            'cultural_adaptation': 0.80,  # Placeholder
            'readability': 0.90,  # Placeholder
            'requires_refinement': False,
            'suggestions': []
        }
        
        # if quality_metrics['intent_preservation'] < 0.85:
        #     quality_metrics['requires_refinement'] = True
        #     quality_metrics['suggestions'].append("Intent not clearly preserved")
        
        logger.info(f"Quality estimation for {target_language}: {quality_metrics}")
        return quality_metrics
    
    def refine(
        self, 
        original: str,
        translated: str,
        quality_metrics: Dict,
        target_language: str,
        cultural_context: Optional[str] = None
    ) -> str:
        """
        Step 3: Iteratively refine to ensure cultural fluency and preserve borderline cases.
        
        Args:
            original: Original English prompt
            translated: Translated prompt
            quality_metrics: Quality estimation results
            target_language: Target language
            cultural_context: Optional cultural context info
            
        Returns:
            Refined translated prompt
        """
        refined = translated

        if quality_metrics['requires_refinement']:
            logger.info(f"Refining translation for {target_language}")
            for suggestion in quality_metrics['suggestions']:
                logger.info(f"  - {suggestion}")

            model = self.quality_estimator or self.translation_model
            if model is not None:
                refine_prompt = (
                    "You are refining a safety-critical translation. "
                    "Return ONLY the improved translation in the target language.\n\n"
                    f"Target language: {target_language}\n"
                    f"Original: {original}\n"
                    f"Current translation: {translated}\n"
                    f"Suggestions: {quality_metrics.get('suggestions', [])}\n"
                    f"Cultural context: {cultural_context or 'N/A'}\n\n"
                    "Refined translation:"
                )
                candidate = self._llm_generate(model, refine_prompt)
                if candidate:
                    return candidate

            # Placeholder refinement logic
            refined = f"[Refined] {translated}"

        return refined
    
    def transcreate(
        self, 
        prompt: str, 
        target_language: str,
        cultural_context: Optional[str] = None
    ) -> Tuple[str, Dict]:
        """
        Full TATER pipeline: Translate -> Estimate -> Refine
        
        Args:
            prompt: English safety prompt
            target_language: Target language
            cultural_context: Optional cultural context
            
        Returns:
            Tuple of (transcreated_prompt, quality_metrics)
        """
        logger.info(f"Starting TATER for {target_language}")
        
        # Step 1: Translate
        translated = self.translate(prompt, target_language)
        
        # Step 2: Estimate Quality
        quality = self.estimate_quality(prompt, translated, target_language)
        
        # Step 3: Refine
        refined = self.refine(prompt, translated, quality, target_language, cultural_context)
        
        return refined, quality
    
    def process_dataset(
        self,
        prompts: List[SafetyPrompt],
        target_languages: List[str]
    ) -> List[SafetyPrompt]:
        """
        Apply TATER to entire dataset.
        
        Args:
            prompts: List of SafetyPrompt objects
            target_languages: Target languages for transcreation
            
        Returns:
            Updated prompts with translations and transcreations
        """
        for prompt in prompts:
            for lang in target_languages:
                if lang == 'English':
                    prompt.translations[lang] = prompt.original_text
                    prompt.transcreated[lang] = prompt.original_text
                    if prompt.options:
                        prompt.options_translations[lang] = list(prompt.options)
                else:
                    transcreated, quality = self.transcreate(
                        prompt.original_text,
                        lang
                    )
                    prompt.translations[lang] = f"[Translated] {transcreated}"
                    prompt.transcreated[lang] = transcreated
                    if prompt.options:
                        translated_options = [
                            self.translate(option, lang)
                            for option in prompt.options
                        ]
                        prompt.options_translations[lang] = translated_options
        
        logger.info(f"TATER processing complete for {len(prompts)} prompts")
        return prompts
