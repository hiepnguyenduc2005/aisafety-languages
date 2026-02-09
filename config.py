"""
Configuration and utility functions for the safety evaluation framework.
"""

import os
import json
from pathlib import Path
from typing import Dict, List, Optional

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
CODE_DIR = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"

# Create directories if they don't exist
DATA_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)

# Language configurations
LANGUAGE_CONFIGS = {
    'English': {
        'code': 'en',
        'script': 'Latin',
        'resource_level': 'High',
        'native_speakers_millions': 1500
    },
    'Vietnamese': {
        'code': 'vi',
        'script': 'Latin',
        'resource_level': 'High',
        'native_speakers_millions': 85
    },
    'Greek': {
        'code': 'el',
        'script': 'Greek',
        'resource_level': 'Mid',
        'native_speakers_millions': 13
    },
    'Amharic': {
        'code': 'am',
        'script': 'Ethiopic',
        'resource_level': 'Mid',
        'native_speakers_millions': 32
    },
    'Tibetan': {
        'code': 'bo',
        'script': 'Tibetan',
        'resource_level': 'Low',
        'native_speakers_millions': 6
    }
}

# Safety domain configurations
SAFETY_DOMAINS = {
    'Crimes': {
        'subdomain_count': 5,
        'severity_range': (1, 3)
    },
    'Fairness': {
        'subdomain_count': 5,
        'severity_range': (0, 3)
    },
    'Privacy': {
        'subdomain_count': 5,
        'severity_range': (1, 3)
    },
    'Physical Harm': {
        'subdomain_count': 5,
        'severity_range': (1, 3)
    },
    'Misuse': {
        'subdomain_count': 5,
        'severity_range': (0, 3)
    }
}

# Model configurations with available options
AVAILABLE_MODELS = {
    'open_source': [
        'meta-llama/Llama-3.1-8B-Instruct',
        'Qwen/Qwen2.5-7B-Instruct',
        'deepseek-ai/DeepSeek-V3',
        'haoranxu/ALMA-7B'
    ],
    'api_based': [
        'gpt-4o',
        'gpt-4',
        'claude-3-opus',
        'claude-3-sonnet'
    ]
}

# Evaluation hyperparameters
EVALUATION_PARAMS = {
    'temperature': 0.7,
    'top_p': 0.9,
    'max_tokens': 512,
    'presence_penalty': 0.0,
    'frequency_penalty': 0.0
}

# Clustering parameters for data curation
CLUSTERING_PARAMS = {
    'n_clusters': 5,  # Match number of domains
    'samples_per_cluster': 50,
    'random_state': 42,
    'n_init': 10,
    'algorithm': 'kmeans++'
}

# Quality estimation thresholds
QUALITY_THRESHOLDS = {
    'intent_preservation': 0.85,
    'cultural_adaptation': 0.75,
    'readability': 0.80
}


class PathManager:
    """Manages file paths for the project."""
    
    @staticmethod
    def get_dataset_path(name: str = "dataset") -> Path:
        """Get path for dataset file."""
        return DATA_DIR / f"{name}.json"
    
    @staticmethod
    def get_results_path(run_id: str, filename: str) -> Path:
        """Get path for results file."""
        run_dir = RESULTS_DIR / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        return run_dir / filename
    
    @staticmethod
    def get_config_path(config_name: str = "config") -> Path:
        """Get path for configuration file."""
        return PROJECT_ROOT / f"{config_name}.json"
    
    @staticmethod
    def ensure_dir(path: Path):
        """Ensure directory exists."""
        path.mkdir(parents=True, exist_ok=True)


class MetricsAggregator:
    """Aggregates evaluation metrics across different dimensions."""
    
    def __init__(self):
        self.metrics = {}
    
    def add_result(
        self,
        model: str,
        language: str,
        domain: str,
        metric_name: str,
        value: float
    ):
        """Add a metric result."""
        key = (model, language, domain, metric_name)
        self.metrics[key] = value
    
    def get_by_model(self, model: str) -> Dict:
        """Get all metrics for a specific model."""
        return {k[1:]: v for k, v in self.metrics.items() if k[0] == model}
    
    def get_by_language(self, language: str) -> Dict:
        """Get all metrics for a specific language."""
        return {k[0:1] + k[2:]: v for k, v in self.metrics.items() if k[1] == language}
    
    def get_average_by_language(self) -> Dict[str, float]:
        """Get average metrics by language."""
        lang_metrics = {}
        for (model, lang, domain, metric), value in self.metrics.items():
            if lang not in lang_metrics:
                lang_metrics[lang] = []
            lang_metrics[lang].append(value)
        
        return {lang: sum(vals) / len(vals) for lang, vals in lang_metrics.items()}
    
    def get_average_by_model(self) -> Dict[str, float]:
        """Get average metrics by model."""
        model_metrics = {}
        for (model, lang, domain, metric), value in self.metrics.items():
            if model not in model_metrics:
                model_metrics[model] = []
            model_metrics[model].append(value)
        
        return {model: sum(vals) / len(vals) for model, vals in model_metrics.items()}


def load_config(config_path: Optional[str] = None) -> Dict:
    """Load configuration from file."""
    if config_path is None:
        config_path = str(PathManager.get_config_path())
    
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            return json.load(f)
    
    return {}


def save_config(config: Dict, config_path: Optional[str] = None):
    """Save configuration to file."""
    if config_path is None:
        config_path = str(PathManager.get_config_path())
    
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
