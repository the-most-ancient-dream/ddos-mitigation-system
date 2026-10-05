"""
Machine Learning module for DDoS attack classification.

The included balanced V2 model uses the DDoSClassifier runtime wrapper.
The CICDataLoader remains for compatibility and unit tests; the supported
training command is src.ml.balanced_dataset_trainer.
"""

from .ml_classifier import DDoSClassifier
from .feature_extractor import FeatureExtractor
from .data_loader import CICDataLoader

__all__ = ['DDoSClassifier', 'FeatureExtractor', 'CICDataLoader']