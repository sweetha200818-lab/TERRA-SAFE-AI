"""
Data loading and preprocessing for TERRA SAFE AI
"""
import pandas as pd
from pathlib import Path
from typing import Tuple, List, Optional, Dict, Any
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib

from src.utils.config import get_config
from src.utils.logger import get_logger

logger = get_logger(__name__)


class DataLoader:
    """Handles loading and initial processing of landslide data"""
    
    def __init__(self, config: Optional[Any] = None):
        self.config = config or get_config()
        self.feature_columns = self.config.features.numerical + self.config.features.categorical
        self.target_column = self.config.features.target[0]
        self.scaler = StandardScaler()
        self.label_encoders = {}
    
    def load_csv(self, file_path: str, **kwargs) -> pd.DataFrame:
        """Load data from CSV file"""
        logger.info(f"Loading data from {file_path}")
        df = pd.read_csv(file_path, **kwargs)
        logger.info(f"Loaded {len(df)} rows, {len(df.columns)} columns")
        return df
    
    def load_geospatial(self, file_path: str, **kwargs) -> 'gpd.GeoDataFrame':
        """Load geospatial data (requires geopandas)"""
        import geopandas as gpd
        logger.info(f"Loading geospatial data from {file_path}")
        gdf = gpd.read_file(file_path, **kwargs)
        logger.info(f"Loaded {len(gdf)} features")
        return gdf
    
    def validate_data(self, df: pd.DataFrame) -> Tuple[bool, List[str]]:
        """Validate data has required columns and no critical issues"""
        errors = []
        required_cols = self.feature_columns + [self.target_column]
        
        missing_cols = set(required_cols) - set(df.columns)
        if missing_cols:
            errors.append(f"Missing required columns: {missing_cols}")
        
        if df.empty:
            errors.append("DataFrame is empty")
        
        # Check for target column
        if self.target_column in df.columns:
            target_unique = df[self.target_column].nunique()
            if target_unique < 2:
                errors.append(f"Target column has only {target_unique} unique value(s)")
        
        # Check for excessive missing values
        missing_pct = df[required_cols].isnull().mean()
        high_missing = missing_pct[missing_pct > 0.5]
        if not high_missing.empty:
            errors.append(f"Columns with >50% missing values: {high_missing.index.tolist()}")
        
        is_valid = len(errors) == 0
        return is_valid, errors
    
    def fill_missing_from_scaler(self, df: pd.DataFrame) -> pd.DataFrame:
        """Impute gaps using statistics captured when the scaler was fitted.

        Used for single live predictions, where a batch median would be undefined.
        """
        df = df.copy()
        numerical = [c for c in self.config.features.numerical if c in df.columns]
        means = getattr(self.scaler, 'mean_', None)
        if means is not None and len(means) == len(numerical):
            for col, mean in zip(numerical, means):
                if df[col].isna().any():
                    df[col] = df[col].fillna(float(mean))
                    logger.warning(f"Filled missing {col} with training mean {float(mean):.4f}")

        for col in self.config.features.categorical:
            if col in df.columns and df[col].isna().any():
                encoder = self.label_encoders.get(col)
                fallback = encoder.classes_[0] if encoder is not None and len(encoder.classes_) else "unknown"
                df[col] = df[col].fillna(fallback)
        return df

    def preprocess_features(self, df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """Preprocess features: handle missing values, encode categorical, scale numerical"""
        df = df.copy()

        # Handle missing values
        for col in self.config.features.numerical:
            if col in df.columns:
                if df[col].isnull().any():
                    median_val = df[col].median()
                    df[col] = df[col].fillna(median_val)
                    logger.warning(f"Filled missing values in {col} with median: {median_val}")
        
        for col in self.config.features.categorical:
            if col in df.columns:
                if df[col].isnull().any():
                    mode_val = df[col].mode()[0] if not df[col].mode().empty else "unknown"
                    df[col] = df[col].fillna(mode_val)
                    logger.warning(f"Filled missing values in {col} with mode: {mode_val}")
        
        # Encode categorical features
        for col in self.config.features.categorical:
            if col in df.columns:
                if fit:
                    le = LabelEncoder()
                    df[col] = le.fit_transform(df[col].astype(str))
                    self.label_encoders[col] = le
                else:
                    if col in self.label_encoders:
                        le = self.label_encoders[col]
                        # Handle unseen categories
                        df[col] = df[col].astype(str).apply(
                            lambda x: le.transform([x])[0] if x in le.classes_ else -1
                        )
        
        # Scale numerical features
        numerical_cols = [c for c in self.config.features.numerical if c in df.columns]
        if numerical_cols:
            if fit:
                df[numerical_cols] = self.scaler.fit_transform(df[numerical_cols])
            else:
                df[numerical_cols] = self.scaler.transform(df[numerical_cols])
        
        return df
    
    def prepare_target(self, df: pd.DataFrame) -> pd.Series:
        """Prepare target variable"""
        target = df[self.target_column].copy()
        
        # Ensure binary classification (0/1)
        if target.dtype == 'object':
            target = target.map({'Low': 0, 'Moderate': 1, 'High': 1, 'Yes': 1, 'No': 0, 
                                 'low': 0, 'moderate': 1, 'high': 1, 'yes': 1, 'no': 0})
        
        # If multiclass, convert to binary (landslide vs no landslide)
        if target.nunique() > 2:
            logger.warning(f"Target has {target.nunique()} classes, converting to binary")
            target = (target > 0).astype(int)
        
        return target
    
    def split_data(self, X: pd.DataFrame, y: pd.Series, 
                   test_size: Optional[float] = None,
                   val_size: Optional[float] = None,
                   random_state: Optional[int] = None,
                   stratify: bool = True) -> Tuple:
        """Split data into train/val/test sets"""
        test_size = test_size or self.config.training.test_size
        val_size = val_size or self.config.training.validation_size
        random_state = random_state or self.config.training.random_state
        
        stratify_param = y if stratify and self.config.training.stratify else None
        
        # First split: train+val vs test
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=stratify_param
        )
        
        # Second split: train vs val
        val_size_adjusted = val_size / (1 - test_size)
        stratify_temp = y_temp if stratify and self.config.training.stratify else None
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_size_adjusted, random_state=random_state, 
            stratify=stratify_temp
        )
        
        logger.info(f"Data split - Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def save_preprocessors(self, path: str):
        """Save fitted preprocessors"""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.scaler, path / "scaler.joblib")
        joblib.dump(self.label_encoders, path / "label_encoders.joblib")
        logger.info(f"Preprocessors saved to {path}")
    
    def load_preprocessors(self, path: str):
        """Load fitted preprocessors"""
        path = Path(path)
        self.scaler = joblib.load(path / "scaler.joblib")
        self.label_encoders = joblib.load(path / "label_encoders.joblib")
        logger.info(f"Preprocessors loaded from {path}")
