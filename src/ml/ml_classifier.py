"""
ML Classifier - DDoS attack detection using Random Forest
Lightweight classifier optimized for real-time packet filtering
"""

import logging
import time
import joblib
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)

logger = logging.getLogger(__name__)


@dataclass
class PredictionResult:
    """Result of ML classification"""
    is_attack: bool
    attack_probability: float
    attack_type: str
    confidence: float
    inference_time_ms: float
    feature_importance: Optional[Dict[str, float]] = None
    
    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            'is_attack': self.is_attack,
            'attack_probability': self.attack_probability,
            'attack_type': self.attack_type,
            'confidence': self.confidence,
            'inference_time_ms': self.inference_time_ms,
        }


@dataclass
class ModelMetrics:
    """Model performance metrics"""
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    confusion_matrix: np.ndarray
    training_time: float
    n_samples: int
    n_features: int
    
    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            'accuracy': self.accuracy,
            'precision': self.precision,
            'recall': self.recall,
            'f1_score': self.f1_score,
            'training_time': self.training_time,
            'n_samples': self.n_samples,
            'n_features': self.n_features,
        }


class DDoSClassifier:
    """
    Random Forest based DDoS attack classifier
    Optimized for real-time inference with low latency
    """
    
    def __init__(self, 
                 model_type: str = 'random_forest',
                 n_estimators: int = 100,
                 max_depth: int = 15,
                 n_jobs: int = -1):
        """
        Initialize the classifier
        
        Args:
            model_type: 'random_forest' or 'gradient_boosting'
            n_estimators: Number of trees
            max_depth: Maximum tree depth
            n_jobs: Number of parallel jobs (-1 = all cores)
        """
        self.model_type = model_type
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.n_jobs = n_jobs
        
        # Initialize model
        if model_type == 'random_forest':
            self.model = RandomForestClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                n_jobs=n_jobs,
                random_state=42,
                min_samples_split=5,
                min_samples_leaf=2,
                class_weight='balanced',
            )
        else:
            self.model = GradientBoostingClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                random_state=42,
                min_samples_split=5,
            )
        
        # Metadata
        self.is_trained = False
        self.feature_names: List[str] = []
        self.scaler = None
        # Per-feature medians from the training split, used only for non-finite
        # incoming feature values. Older saved models have no imputer metadata.
        self.imputation_values: Optional[np.ndarray] = None
        self.metrics: Optional[ModelMetrics] = None
        self.training_date: Optional[str] = None
        self.training_accuracy: Optional[float] = None
        
        # Attack type mapping for multi-class
        self.attack_types = [
            'BENIGN', 'DrDoS_UDP', 'DrDoS_DNS', 'DrDoS_LDAP',
            'DrDoS_MSSQL', 'DrDoS_NTP', 'Syn', 'HTTP_Flood'
        ]
        
        # Inference stats
        self.total_predictions = 0
        self.total_inference_time = 0.0
    
    def train(self, 
              X_train: np.ndarray, 
              y_train: np.ndarray,
              X_val: Optional[np.ndarray] = None,
              y_val: Optional[np.ndarray] = None,
              feature_names: Optional[List[str]] = None) -> ModelMetrics:
        """
        Train the classifier
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features (optional)
            y_val: Validation labels (optional)
            feature_names: List of feature names
            
        Returns:
            ModelMetrics object with training results
        """
        logger.info(f"Training {self.model_type} classifier...")
        logger.info(f"Training samples: {len(X_train)}, Features: {X_train.shape[1]}")
        
        start_time = time.time()
        
        # Train model
        self.model.fit(X_train, y_train)

        self.training_accuracy = accuracy_score(y_train, self.model.predict(X_train))
        logger.info(f"Training accuracy: {self.training_accuracy:.4f}")
        
        training_time = time.time() - start_time
        logger.info(f"Training completed in {training_time:.2f}s")
        
        # Store feature names
        if feature_names:
            self.feature_names = feature_names
        
        # Evaluate on validation set or training set
        eval_X = X_val if X_val is not None else X_train
        eval_y = y_val if y_val is not None else y_train
        
        y_pred = self.model.predict(eval_X)
        
        # Calculate metrics
        self.metrics = ModelMetrics(
            accuracy=accuracy_score(eval_y, y_pred),
            precision=precision_score(eval_y, y_pred, average='weighted', zero_division=0),
            recall=recall_score(eval_y, y_pred, average='weighted', zero_division=0),
            f1_score=f1_score(eval_y, y_pred, average='weighted', zero_division=0),
            confusion_matrix=confusion_matrix(eval_y, y_pred),
            training_time=training_time,
            n_samples=len(X_train),
            n_features=X_train.shape[1],
        )
        
        evaluation_name = 'Validation' if X_val is not None and y_val is not None else 'Training'
        logger.info(f"{evaluation_name} accuracy: {self.metrics.accuracy:.4f}")
        logger.info(f"{evaluation_name} precision: {self.metrics.precision:.4f}")
        logger.info(f"{evaluation_name} recall: {self.metrics.recall:.4f}")
        logger.info(f"{evaluation_name} F1 score: {self.metrics.f1_score:.4f}")
        
        self.is_trained = True
        self.training_date = time.strftime('%Y-%m-%d %H:%M:%S')
        
        return self.metrics
    
    def predict(self, X: np.ndarray) -> PredictionResult:
        """
        Make prediction on input features
        
        Args:
            X: Feature array of shape (1, n_features) or (n_features,)
            
        Returns:
            PredictionResult object
        """
        if not self.is_trained:
            logger.warning("Model not trained, returning default prediction")
            return PredictionResult(
                is_attack=False,
                attack_probability=0.0,
                attack_type='UNKNOWN',
                confidence=0.0,
                inference_time_ms=0.0,
            )
        
        # Ensure 2D array
        if X.ndim == 1:
            X = X.reshape(1, -1)
        
        start_time = time.time()
        
        # Get prediction and probabilities
        prediction = self.model.predict(X)[0]
        probabilities = self.model.predict_proba(X)[0]
        
        inference_time = (time.time() - start_time) * 1000  # ms
        
        # Update stats
        self.total_predictions += 1
        self.total_inference_time += inference_time
        
        # Label 0 is BENIGN in both the binary and encoded multiclass pipelines.
        classes = list(getattr(self.model, 'classes_', range(len(probabilities))))
        is_attack = prediction != 0
        attack_prob = sum(
            probability
            for label, probability in zip(classes, probabilities)
            if label != 0
        )
        confidence = max(probabilities) * 100
        
        # For binary classification, infer attack type from features
        if is_attack:
            attack_type = self._infer_attack_type(
                self._unscale_features(X[0]), self.feature_names
            )
        else:
            attack_type = 'BENIGN'
        
        return PredictionResult(
            is_attack=is_attack,
            attack_probability=float(attack_prob),
            attack_type=attack_type,
            confidence=float(confidence),
            inference_time_ms=inference_time,
        )
    
    def predict_batch(self, X: np.ndarray) -> List[PredictionResult]:
        """
        Make predictions on a batch of samples
        
        Args:
            X: Feature array of shape (n_samples, n_features)
            
        Returns:
            List of PredictionResult objects
        """
        if not self.is_trained:
            return []
        
        start_time = time.time()
        
        predictions = self.model.predict(X)
        probabilities = self.model.predict_proba(X)
        
        inference_time = (time.time() - start_time) * 1000 / len(X)
        
        results = []
        for i, (pred, probs) in enumerate(zip(predictions, probabilities)):
            classes = list(getattr(self.model, 'classes_', range(len(probs))))
            is_attack = pred != 0
            attack_prob = sum(
                probability
                for label, probability in zip(classes, probs)
                if label != 0
            )
            
            results.append(PredictionResult(
                is_attack=is_attack,
                attack_probability=float(attack_prob),
                attack_type=(
                    self._infer_attack_type(
                        self._unscale_features(X[i]), self.feature_names
                    )
                    if is_attack else 'BENIGN'
                ),
                confidence=float(max(probs) * 100),
                inference_time_ms=inference_time,
            ))
        
        return results
    
    def _unscale_features(self, features: np.ndarray) -> np.ndarray:
        """Recover feature magnitudes for the heuristic attack-type label."""
        if self.scaler is None:
            return features

        expected_count = getattr(self.scaler, 'n_features_in_', len(features))
        if expected_count != len(features):
            return features

        try:
            return self.scaler.inverse_transform(features.reshape(1, -1))[0]
        except Exception:
            return features

    def _infer_attack_type(
        self,
        features: np.ndarray,
        feature_names: Optional[List[str]] = None,
    ) -> str:
        """
        Infer attack type from features (heuristic for binary classification)
        
        Args:
            features: Single sample features
            
        Returns:
            Inferred attack type string
        """
        try:
            names = feature_names or self.feature_names
            positions = {name: index for index, name in enumerate(names)}

            def get_feature(name: str) -> float:
                index = positions.get(name)
                return float(features[index]) if index is not None else 0.0

            syn_count = get_feature('SYN Flag Count')
            ack_count = get_feature('ACK Flag Count')
            packets_per_sec = get_feature('Flow Packets/s')
            bytes_per_sec = get_feature('Flow Bytes/s')
            
            # Heuristic classification
            if syn_count > 100 and ack_count < 10:
                return 'SYN_Flood'
            elif packets_per_sec > 10000 and bytes_per_sec > 1000000:
                return 'UDP_Flood'
            elif packets_per_sec > 5000:
                return 'DrDoS_UDP'
            else:
                return 'DDoS_Generic'
                
        except Exception:
            return 'DDoS_Generic'
    
    def get_feature_importance(self, top_n: int = 20) -> Dict[str, float]:
        """
        Get feature importance from the trained model
        
        Args:
            top_n: Number of top features to return
            
        Returns:
            Dictionary of feature name -> importance
        """
        if not self.is_trained:
            return {}
        
        importances = self.model.feature_importances_
        
        if self.feature_names and len(self.feature_names) == len(importances):
            importance_dict = dict(zip(self.feature_names, importances))
        else:
            importance_dict = {f'feature_{i}': imp for i, imp in enumerate(importances)}
        
        # Sort by importance and return top N
        sorted_importance = dict(
            sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)[:top_n]
        )
        
        return sorted_importance
    
    def save(self, filepath: str) -> bool:
        """
        Save trained model to file
        
        Args:
            filepath: Path to save model
            
        Returns:
            True if successful
        """
        if not self.is_trained:
            logger.error("Cannot save untrained model")
            return False
        
        try:
            save_dict = {
                'model': self.model,
                'model_type': self.model_type,
                'feature_names': self.feature_names,
                'scaler': self.scaler,
                'imputation_values': self.imputation_values,
                'metrics': self.metrics,
                'training_date': self.training_date,
                'training_accuracy': self.training_accuracy,
                'n_estimators': self.n_estimators,
                'max_depth': self.max_depth,
            }
            
            Path(filepath).parent.mkdir(parents=True, exist_ok=True)
            joblib.dump(save_dict, filepath)
            logger.info(f"Model saved to {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save model: {e}")
            return False
    
    def load(self, filepath: str) -> bool:
        """
        Load trained model from file
        
        Args:
            filepath: Path to model file
            
        Returns:
            True if successful
        """
        try:
            if not Path(filepath).exists():
                logger.error(f"Model file not found: {filepath}")
                return False
            
            save_dict = joblib.load(filepath)
            
            self.model = save_dict['model']
            self.model_type = save_dict.get('model_type', 'random_forest')
            self.feature_names = save_dict.get('feature_names', [])
            self.scaler = save_dict.get('scaler')
            imputation_values = save_dict.get('imputation_values')
            self.imputation_values = (
                np.asarray(imputation_values, dtype=np.float64)
                if imputation_values is not None else None
            )
            self.metrics = save_dict.get('metrics')
            self.training_date = save_dict.get('training_date')
            self.training_accuracy = save_dict.get('training_accuracy')
            self.n_estimators = save_dict.get('n_estimators', 100)
            self.max_depth = save_dict.get('max_depth', 15)
            self.is_trained = True
            
            logger.info(f"Model loaded from {filepath}")
            logger.info(f"Training date: {self.training_date}")
            if self.training_accuracy is not None:
                logger.info(f"Training accuracy: {self.training_accuracy:.4f}")
            if self.metrics:
                logger.info(f"Validation accuracy: {self.metrics.accuracy:.4f}")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            return False
    
    def get_stats(self) -> Dict:
        """Get classifier statistics"""
        avg_inference = (
            self.total_inference_time / self.total_predictions 
            if self.total_predictions > 0 else 0
        )
        
        return {
            'is_trained': self.is_trained,
            'model_type': self.model_type,
            'n_estimators': self.n_estimators,
            'max_depth': self.max_depth,
            'n_features': len(self.feature_names),
            'training_date': self.training_date,
            'training_accuracy': self.training_accuracy,
            'total_predictions': self.total_predictions,
            'avg_inference_ms': avg_inference,
            'metrics': self.metrics.to_dict() if self.metrics else None,
        }
