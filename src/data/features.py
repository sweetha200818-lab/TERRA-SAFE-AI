"""
Feature engineering for TERRA SAFE AI
"""
import pandas as pd
import numpy as np
from typing import List, Optional, Any
from sklearn.base import BaseEstimator, TransformerMixin

from src.utils.config import get_config
from src.utils.logger import get_logger

logger = get_logger(__name__)


class FeatureEngineer(BaseEstimator, TransformerMixin):
    """Create derived features for landslide prediction"""
    
    def __init__(self, config: Optional[Any] = None):
        self.config = config or get_config()
        self.feature_names_ = None
    
    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> 'FeatureEngineer':
        self.feature_names_ = X.columns.tolist()
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        
        # Topographic features
        if 'slope_degrees' in X.columns:
            X['slope_radians'] = np.deg2rad(X['slope_degrees'])
            X['slope_sin'] = np.sin(X['slope_radians'])
            X['slope_cos'] = np.cos(X['slope_radians'])
            X['slope_tan'] = np.tan(X['slope_radians'])
            X['slope_squared'] = X['slope_degrees'] ** 2
        
        if 'elevation_m' in X.columns:
            X['elevation_normalized'] = X['elevation_m'] / X['elevation_m'].max()
            X['elevation_squared'] = X['elevation_m'] ** 2
        
        # Combined topographic indices
        if 'slope_degrees' in X.columns and 'elevation_m' in X.columns:
            X['topographic_index'] = X['slope_degrees'] * X['elevation_m'] / 1000
            X['relief_ratio'] = X['slope_degrees'] / (X['elevation_m'] / 1000 + 1)
        
        # Hydrological features
        if 'rainfall_mm' in X.columns and 'soil_moisture' in X.columns:
            X['rainfall_moisture_interaction'] = X['rainfall_mm'] * X['soil_moisture']
            X['saturation_index'] = X['soil_moisture'] * (X['rainfall_mm'] / 100)
        
        if 'rainfall_mm' in X.columns:
            X['rainfall_log'] = np.log1p(X['rainfall_mm'])
            X['rainfall_squared'] = X['rainfall_mm'] ** 2
            X['heavy_rainfall'] = (X['rainfall_mm'] > 150).astype(int)
            X['extreme_rainfall'] = (X['rainfall_mm'] > 300).astype(int)
        
        if 'soil_moisture' in X.columns:
            X['soil_moisture_squared'] = X['soil_moisture'] ** 2
            X['high_moisture'] = (X['soil_moisture'] > 0.6).astype(int)
            X['saturated'] = (X['soil_moisture'] > 0.8).astype(int)
        
        # Climatic features
        if 'temperature_c' in X.columns and 'humidity_percent' in X.columns:
            X['heat_index'] = X['temperature_c'] * X['humidity_percent'] / 100
            X['dew_point_proxy'] = X['temperature_c'] - (100 - X['humidity_percent']) / 5
        
        if 'humidity_percent' in X.columns:
            X['high_humidity'] = (X['humidity_percent'] > 80).astype(int)
        
        # Distance features
        if 'distance_to_road_km' in X.columns:
            X['road_proximity'] = 1 / (X['distance_to_road_km'] + 0.1)
            X['near_road'] = (X['distance_to_road_km'] < 1.0).astype(int)
            X['road_log'] = np.log1p(X['distance_to_road_km'])
        
        # Land cover features (if encoded)
        if 'land_cover' in X.columns:
            # Create risk-based land cover grouping
            high_risk_covers = ['barren', 'agriculture']
            medium_risk_covers = ['grassland', 'urban']
            low_risk_covers = ['forest', 'wetland']
            
            if X['land_cover'].dtype in ['object', 'string']:
                X['land_cover_risk'] = X['land_cover'].map({
                    **{c: 2 for c in high_risk_covers},
                    **{c: 1 for c in medium_risk_covers},
                    **{c: 0 for c in low_risk_covers}
                }).fillna(1).astype(int)
            else:
                # Assume already encoded
                pass
        
        # Multi-factor risk indices
        if all(c in X.columns for c in ['slope_degrees', 'rainfall_mm', 'soil_moisture']):
            X['combined_risk_index'] = (
                0.4 * (X['slope_degrees'] / 60) +
                0.3 * (X['rainfall_mm'] / 500) +
                0.3 * X['soil_moisture']
            )
        
        if all(c in X.columns for c in ['slope_degrees', 'elevation_m', 'rainfall_mm']):
            X['geomorphic_risk'] = (
                X['slope_degrees'] / 60 * 
                X['elevation_m'] / 3000 * 
                X['rainfall_mm'] / 500
            ) * 1000
        
        logger.debug(f"Feature engineering: {len(X.columns)} features created")
        return X
    
    def get_feature_names_out(self, input_features: Optional[List[str]] = None) -> List[str]:
        if self.feature_names_ is None:
            return input_features or []
        
        # Create dummy dataframe to get output feature names
        dummy = pd.DataFrame(columns=self.feature_names_)
        dummy = self.transform(dummy)
        return dummy.columns.tolist()


class FeatureSelector(BaseEstimator, TransformerMixin):
    """Select features based on importance or correlation"""
    
    def __init__(self, method: str = 'importance', threshold: float = 0.01, 
                 max_features: Optional[int] = None):
        self.method = method
        self.threshold = threshold
        self.max_features = max_features
        self.selected_features_ = None
        self.feature_importances_ = None
    
    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> 'FeatureSelector':
        if self.method == 'importance' and y is not None:
            from sklearn.ensemble import RandomForestClassifier
            rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
            rf.fit(X, y)
            self.feature_importances_ = pd.Series(rf.feature_importances_, index=X.columns)
            self.selected_features_ = self.feature_importances_[
                self.feature_importances_ >= self.threshold
            ].index.tolist()
        elif self.method == 'correlation':
            corr_matrix = X.corr().abs()
            upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
            to_drop = [col for col in upper.columns if any(upper[col] > 0.95)]
            self.selected_features_ = [c for c in X.columns if c not in to_drop]
        else:
            self.selected_features_ = X.columns.tolist()
        
        if self.max_features and len(self.selected_features_) > self.max_features:
            if self.feature_importances_ is not None:
                self.selected_features_ = self.feature_importances_.nlargest(self.max_features).index.tolist()
            else:
                self.selected_features_ = self.selected_features_[:self.max_features]
        
        logger.info(f"Feature selection ({self.method}): {len(self.selected_features_)} features selected")
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if self.selected_features_ is None:
            return X
        return X[self.selected_features_]
    
    def get_feature_names_out(self, input_features: Optional[List[str]] = None) -> List[str]:
        return self.selected_features_ or input_features or []


def create_pipeline():
    """Create the full feature engineering pipeline"""
    from sklearn.pipeline import Pipeline
    
    return Pipeline([
        ('engineer', FeatureEngineer()),
        ('selector', FeatureSelector(method='importance', threshold=0.005, max_features=50))
    ])