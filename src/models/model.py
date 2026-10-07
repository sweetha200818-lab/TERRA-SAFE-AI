"""
ML Models for TERRA SAFE AI - Random Forest, XGBoost, and Ensemble
"""
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, List, Tuple
from pathlib import Path
import joblib

from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier

from src.utils.config import get_config
from src.utils.logger import get_logger

logger = get_logger(__name__)


class LandslideModel:
    """Base class for landslide prediction models"""
    
    def __init__(self, name: str, config: Optional[Any] = None):
        self.name = name
        self.config = config or get_config()
        self.model = None
        self.is_fitted = False
        self.feature_names = None
        self.training_history = {}
    
    def fit(self, X: pd.DataFrame, y: pd.Series, 
            X_val: Optional[pd.DataFrame] = None, 
            y_val: Optional[pd.Series] = None) -> 'LandslideModel':
        raise NotImplementedError
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError(f"{self.name} model is not fitted")
        return self.model.predict(X)
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError(f"{self.name} model is not fitted")
        return self.model.predict_proba(X)[:, 1]
    
    def get_feature_importance(self) -> pd.Series:
        if not self.is_fitted:
            raise ValueError(f"{self.name} model is not fitted")
        raise NotImplementedError
    
    def save(self, path: str):
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.model, path / f"{self.name}.joblib")
        joblib.dump({
            'feature_names': self.feature_names,
            'training_history': self.training_history
        }, path / f"{self.name}_meta.joblib")
        logger.info(f"{self.name} model saved to {path}")
    
    def load(self, path: str):
        path = Path(path)
        self.model = joblib.load(path / f"{self.name}.joblib")
        meta = joblib.load(path / f"{self.name}_meta.joblib")
        self.feature_names = meta['feature_names']
        self.training_history = meta['training_history']
        self.is_fitted = True
        logger.info(f"{self.name} model loaded from {path}")


class RandomForestModel(LandslideModel):
    """Random Forest model for landslide prediction"""
    
    def __init__(self, config: Optional[Any] = None):
        super().__init__("random_forest", config)
        rf_config = self.config.models.random_forest
        self.model = RandomForestClassifier(
            n_estimators=rf_config.n_estimators,
            max_depth=rf_config.max_depth,
            min_samples_split=rf_config.min_samples_split,
            min_samples_leaf=rf_config.min_samples_leaf,
            max_features=rf_config.max_features,
            random_state=rf_config.random_state,
            n_jobs=rf_config.n_jobs,
            class_weight=rf_config.class_weight,
            verbose=0
        )
    
    def fit(self, X: pd.DataFrame, y: pd.Series,
            X_val: Optional[pd.DataFrame] = None,
            y_val: Optional[pd.Series] = None) -> 'RandomForestModel':
        self.feature_names = X.columns.tolist()
        
        logger.info(f"Training Random Forest with {len(X)} samples, {len(X.columns)} features")
        
        self.model.fit(X, y)
        self.is_fitted = True
        
        # Training metrics
        train_auc = roc_auc_score(y, self.predict_proba(X))
        self.training_history['train_auc'] = train_auc
        
        if X_val is not None and y_val is not None:
            val_auc = roc_auc_score(y_val, self.predict_proba(X_val))
            self.training_history['val_auc'] = val_auc
            logger.info(f"Random Forest - Train AUC: {train_auc:.4f}, Val AUC: {val_auc:.4f}")
        else:
            logger.info(f"Random Forest - Train AUC: {train_auc:.4f}")
        
        # Cross-validation
        if self.config.training.cross_validation.enabled:
            cv = StratifiedKFold(
                n_splits=self.config.training.cross_validation.n_splits,
                shuffle=True,
                random_state=self.config.training.random_state
            )
            cv_scores = cross_val_score(
                self.model, X, y, cv=cv, scoring=self.config.training.cross_validation.scoring, n_jobs=-1
            )
            self.training_history['cv_scores'] = cv_scores.tolist()
            self.training_history['cv_mean'] = cv_scores.mean()
            self.training_history['cv_std'] = cv_scores.std()
            logger.info(f"Random Forest CV AUC: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
        
        return self
    
    def get_feature_importance(self) -> pd.Series:
        if not self.is_fitted:
            raise ValueError("Model is not fitted")
        return pd.Series(
            self.model.feature_importances_,
            index=self.feature_names
        ).sort_values(ascending=False)


class XGBoostModel(LandslideModel):
    """XGBoost model for landslide prediction"""
    
    def __init__(self, config: Optional[Any] = None):
        super().__init__("xgboost", config)
        xgb_config = self.config.models.xgboost
        self.model = XGBClassifier(
            n_estimators=xgb_config.n_estimators,
            max_depth=xgb_config.max_depth,
            learning_rate=xgb_config.learning_rate,
            subsample=xgb_config.subsample,
            colsample_bytree=xgb_config.colsample_bytree,
            random_state=xgb_config.random_state,
            n_jobs=xgb_config.n_jobs,
            scale_pos_weight=xgb_config.scale_pos_weight,
            eval_metric=xgb_config.eval_metric,
            verbosity=0
        )
    
    def fit(self, X: pd.DataFrame, y: pd.Series,
            X_val: Optional[pd.DataFrame] = None,
            y_val: Optional[pd.Series] = None) -> 'XGBoostModel':
        self.feature_names = X.columns.tolist()
        
        logger.info(f"Training XGBoost with {len(X)} samples, {len(X.columns)} features")
        
        # Prepare eval set
        eval_set = []
        if X_val is not None and y_val is not None:
            eval_set = [(X_val, y_val)]
        
        self.model.fit(
            X, y,
            eval_set=eval_set,
            verbose=False
        )
        self.is_fitted = True
        
        # Training metrics
        train_auc = roc_auc_score(y, self.predict_proba(X))
        self.training_history['train_auc'] = train_auc
        
        if X_val is not None and y_val is not None:
            val_auc = roc_auc_score(y_val, self.predict_proba(X_val))
            self.training_history['val_auc'] = val_auc
            logger.info(f"XGBoost - Train AUC: {train_auc:.4f}, Val AUC: {val_auc:.4f}")
        else:
            logger.info(f"XGBoost - Train AUC: {train_auc:.4f}")
        
        # Cross-validation
        if self.config.training.cross_validation.enabled:
            cv = StratifiedKFold(
                n_splits=self.config.training.cross_validation.n_splits,
                shuffle=True,
                random_state=self.config.training.random_state
            )
            cv_scores = cross_val_score(
                self.model, X, y, cv=cv, scoring=self.config.training.cross_validation.scoring, n_jobs=-1
            )
            self.training_history['cv_scores'] = cv_scores.tolist()
            self.training_history['cv_mean'] = cv_scores.mean()
            self.training_history['cv_std'] = cv_scores.std()
            logger.info(f"XGBoost CV AUC: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
        
        return self
    
    def get_feature_importance(self) -> pd.Series:
        if not self.is_fitted:
            raise ValueError("Model is not fitted")
        return pd.Series(
            self.model.feature_importances_,
            index=self.feature_names
        ).sort_values(ascending=False)


class EnsembleModel(LandslideModel):
    """Ensemble model combining Random Forest and XGBoost"""
    
    def __init__(self, config: Optional[Any] = None):
        super().__init__("ensemble", config)
        self.rf_model = RandomForestModel(config)
        self.xgb_model = XGBoostModel(config)
        self.weights = self.config.models.ensemble.weights
        self.voting = self.config.models.ensemble.voting
    
    def fit(self, X: pd.DataFrame, y: pd.Series,
            X_val: Optional[pd.DataFrame] = None,
            y_val: Optional[pd.Series] = None) -> 'EnsembleModel':
        self.feature_names = X.columns.tolist()
        
        logger.info("Training Ensemble (Random Forest + XGBoost)")
        
        # Train individual models
        self.rf_model.fit(X, y, X_val, y_val)
        self.xgb_model.fit(X, y, X_val, y_val)
        
        # Create voting classifier
        self.model = VotingClassifier(
            estimators=[
                ('rf', self.rf_model.model),
                ('xgb', self.xgb_model.model)
            ],
            voting=self.voting,
            weights=self.weights,
            n_jobs=-1
        )
        
        self.model.fit(X, y)
        self.is_fitted = True
        
        # Training metrics
        train_auc = roc_auc_score(y, self.predict_proba(X))
        self.training_history['train_auc'] = train_auc
        self.training_history['rf_auc'] = self.rf_model.training_history.get('train_auc')
        self.training_history['xgb_auc'] = self.xgb_model.training_history.get('train_auc')
        
        if X_val is not None and y_val is not None:
            val_auc = roc_auc_score(y_val, self.predict_proba(X_val))
            self.training_history['val_auc'] = val_auc
            logger.info(f"Ensemble - Train AUC: {train_auc:.4f}, Val AUC: {val_auc:.4f}")
        else:
            logger.info(f"Ensemble - Train AUC: {train_auc:.4f}")
        
        return self
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError("Model is not fitted")
        # Use weighted average of individual model predictions
        rf_pred = self.rf_model.predict(X)
        xgb_pred = self.xgb_model.predict(X)
        # Majority vote with weights
        combined = (self.weights[0] * rf_pred + self.weights[1] * xgb_pred) / sum(self.weights)
        return (combined >= 0.5).astype(int)
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError("Model is not fitted")
        # Weighted average of probabilities
        rf_proba = self.rf_model.predict_proba(X)
        xgb_proba = self.xgb_model.predict_proba(X)
        combined = (self.weights[0] * rf_proba + self.weights[1] * xgb_proba) / sum(self.weights)
        return combined
    
    def get_feature_importance(self) -> pd.Series:
        """Average feature importance from both models"""
        if not self.is_fitted:
            raise ValueError("Model is not fitted")
        
        rf_imp = self.rf_model.get_feature_importance()
        xgb_imp = self.xgb_model.get_feature_importance()
        
        # Weighted average
        combined = (self.weights[0] * rf_imp + self.weights[1] * xgb_imp) / sum(self.weights)
        return combined.sort_values(ascending=False)
    
    def save(self, path: str):
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        self.rf_model.save(path)
        self.xgb_model.save(path)
        joblib.dump({
            'weights': self.weights,
            'voting': self.voting,
            'feature_names': self.feature_names,
            'training_history': self.training_history
        }, path / "ensemble_meta.joblib")
        logger.info(f"Ensemble model saved to {path}")
    
    def load(self, path: str):
        path = Path(path)
        self.rf_model.load(path)
        self.xgb_model.load(path)
        meta = joblib.load(path / "ensemble_meta.joblib")
        self.weights = meta['weights']
        self.voting = meta['voting']
        self.feature_names = meta['feature_names']
        self.training_history = meta['training_history']
        self.is_fitted = True
        logger.info(f"Ensemble model loaded from {path}")


def create_model(model_type: str, config: Optional[Any] = None) -> LandslideModel:
    """Factory function to create models"""
    models = {
        'random_forest': RandomForestModel,
        'rf': RandomForestModel,
        'xgboost': XGBoostModel,
        'xgb': XGBoostModel,
        'ensemble': EnsembleModel
    }
    
    model_type = model_type.lower()
    if model_type not in models:
        raise ValueError(f"Unknown model type: {model_type}. Available: {list(models.keys())}")
    
    return models[model_type](config)