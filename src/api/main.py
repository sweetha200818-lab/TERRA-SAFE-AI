"""
Risk Prediction API for TERRA SAFE AI
"""
from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel, Field, validator
from typing import List, Optional, Dict, Any
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
import json
from contextlib import asynccontextmanager

from src.utils.config import get_config
from src.utils.logger import get_logger, setup_logger
from src.models.model import create_model, EnsembleModel
from src.data.loader import DataLoader
from src.data.realtime import RealTimeDataFetcher

logger = get_logger(__name__)

# Global model instance
model_instance = None
data_loader = None


class PredictionInput(BaseModel):
    """Input schema for single prediction"""
    rainfall_mm: float = Field(..., ge=0, le=1000, description="Rainfall in mm")
    soil_moisture: float = Field(..., ge=0, le=1, description="Soil moisture (0-1)")
    temperature_c: float = Field(..., ge=-20, le=50, description="Temperature in Celsius")
    humidity_percent: float = Field(..., ge=0, le=100, description="Humidity percentage")
    slope_degrees: float = Field(..., ge=0, le=90, description="Slope in degrees")
    elevation_m: float = Field(..., ge=0, le=6000, description="Elevation in meters")
    distance_to_road_km: float = Field(..., ge=0, le=100, description="Distance to road in km")
    land_cover: str = Field(..., description="Land cover type")
    
    @validator('land_cover')
    def validate_land_cover(cls, v):
        valid_covers = ['forest', 'grassland', 'agriculture', 'urban', 'barren', 'wetland']
        if v.lower() not in valid_covers:
            raise ValueError(f"Invalid land_cover. Must be one of: {valid_covers}")
        return v.lower()


class BatchPredictionInput(BaseModel):
    """Input schema for batch predictions"""
    predictions: List[PredictionInput] = Field(..., min_items=1, max_items=1000)


class PredictionOutput(BaseModel):
    """Output schema for prediction"""
    probability: float = Field(..., description="Landslide probability (0-1)")
    risk_level: str = Field(..., description="Risk classification: LOW, MODERATE, HIGH")
    risk_score: float = Field(..., description="Risk score (0-100)")
    model_version: str = Field(..., description="Model version used")


class BatchPredictionOutput(BaseModel):
    """Output schema for batch predictions"""
    predictions: List[PredictionOutput]
    summary: Dict[str, Any]


class LiveLocationInput(BaseModel):
    """A coordinate whose conditions are fetched in real time"""
    latitude: float = Field(..., ge=-90, le=90, description="Latitude in decimal degrees")
    longitude: float = Field(..., ge=-180, le=180, description="Longitude in decimal degrees")
    refresh: bool = Field(False, description="Bypass the local cache and re-download live data")


class LiveConditionsOutput(BaseModel):
    """Real-time environmental conditions fetched for a coordinate"""
    latitude: float
    longitude: float
    observed_at: Optional[str] = None
    features: Dict[str, Any]
    sources: Dict[str, str]


class LivePredictionOutput(PredictionOutput):
    """Prediction built entirely from real-time conditions"""
    latitude: float
    longitude: float
    observed_at: Optional[str] = None
    conditions: Dict[str, Any]
    sources: Dict[str, str]


class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    model_loaded: bool
    model_version: Optional[str] = None
    api_version: str


def load_model_and_preprocessors():
    """Load the trained model and preprocessors"""
    global model_instance, data_loader
    
    config = get_config()
    model_path = Path(config.paths.models) / "ensemble"
    preproc_path = Path(config.paths.models)
    
    if not model_path.exists():
        logger.warning(f"Model not found at {model_path}. API will not be able to make predictions.")
        return False
    
    try:
        # Load ensemble model
        model_instance = EnsembleModel(config)
        model_instance.load(str(model_path))
        
        # Load preprocessors from main models directory
        data_loader = DataLoader(config)
        data_loader.load_preprocessors(str(preproc_path))
        
        logger.info("Model and preprocessors loaded successfully")
        return True
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        return False


def predict_risk(input_data: pd.DataFrame) -> List[Dict]:
    """Make predictions on input data"""
    if model_instance is None or not model_instance.is_fitted:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    # Impute anything the live sources could not return, then preprocess
    prepared = data_loader.fill_missing_from_scaler(input_data)
    processed = data_loader.preprocess_features(prepared, fit=False)
    
    # Get feature names from model
    feature_names = model_instance.feature_names
    if feature_names:
        # Ensure all features are present
        for feat in feature_names:
            if feat not in processed.columns:
                processed[feat] = 0
        processed = processed[feature_names]
    
    # Predict
    probabilities = model_instance.predict_proba(processed)
    predictions = model_instance.predict(processed)
    
    # Classify risk
    config = get_config()
    results = []
    for prob, pred in zip(probabilities, predictions):
        if prob <= config.risk_thresholds.low_max:
            risk_level = "LOW"
        elif prob <= config.risk_thresholds.moderate_max:
            risk_level = "MODERATE"
        else:
            risk_level = "HIGH"
        
        results.append({
            "probability": float(prob),
            "risk_level": risk_level,
            "risk_score": round(float(prob) * 100, 1),
            "model_version": "1.0.0"
        })
    
    return results


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler"""
    # Startup
    setup_logger()
    load_model_and_preprocessors()
    yield
    # Shutdown (if needed)
    pass


# Create FastAPI app
config = get_config()
app = FastAPI(
    title=config.api.title,
    description=config.api.description,
    version=config.api.version,
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy" if model_instance and model_instance.is_fitted else "degraded",
        model_loaded=model_instance is not None and model_instance.is_fitted,
        model_version="1.0.0" if model_instance and model_instance.is_fitted else None,
        api_version=config.api.version
    )


@app.post("/predict", response_model=PredictionOutput)
async def predict_single(input_data: PredictionInput):
    """Predict landslide risk for a single location"""
    df = pd.DataFrame([input_data.dict()])
    results = predict_risk(df)
    return PredictionOutput(**results[0])


@app.post("/predict/batch", response_model=BatchPredictionOutput)
async def predict_batch(input_data: BatchPredictionInput):
    """Predict landslide risk for multiple locations"""
    df = pd.DataFrame([item.dict() for item in input_data.predictions])
    results = predict_risk(df)
    
    # Summary statistics
    probs = [r["probability"] for r in results]
    risk_levels = [r["risk_level"] for r in results]
    
    summary = {
        "total_predictions": len(results),
        "mean_probability": round(np.mean(probs), 4),
        "max_probability": round(np.max(probs), 4),
        "min_probability": round(np.min(probs), 4),
        "risk_distribution": {
            "LOW": risk_levels.count("LOW"),
            "MODERATE": risk_levels.count("MODERATE"),
            "HIGH": risk_levels.count("HIGH")
        },
        "high_risk_percentage": round(risk_levels.count("HIGH") / len(results) * 100, 1)
    }
    
    return BatchPredictionOutput(
        predictions=[PredictionOutput(**r) for r in results],
        summary=summary
    )


def _fetch_live_conditions(latitude: float, longitude: float, refresh: bool = False) -> Dict[str, Any]:
    """Fetch real-time conditions for a coordinate (blocking, run in threadpool)."""
    fetcher = RealTimeDataFetcher(get_config(), refresh=refresh)
    try:
        return fetcher.fetch_live_conditions(latitude, longitude)
    except Exception as exc:
        logger.error(f"Real-time fetch failed for ({latitude}, {longitude}): {exc}")
        raise HTTPException(status_code=502, detail=f"Real-time data source unavailable: {exc}")


@app.get("/data/live", response_model=LiveConditionsOutput)
def live_conditions(latitude: float, longitude: float, refresh: bool = False):
    """Fetch the real-time environmental conditions used for prediction"""
    payload = _fetch_live_conditions(latitude, longitude, refresh)
    return LiveConditionsOutput(
        latitude=payload["latitude"],
        longitude=payload["longitude"],
        observed_at=payload.get("observed_at"),
        features=payload["features"],
        sources=payload["sources"],
    )


@app.post("/predict/live", response_model=LivePredictionOutput)
def predict_live(location: LiveLocationInput):
    """Predict landslide risk from live conditions at a coordinate"""
    payload = _fetch_live_conditions(location.latitude, location.longitude, location.refresh)
    df = pd.DataFrame([payload["features"]])
    results = predict_risk(df)
    result = results[0]
    return LivePredictionOutput(
        **result,
        latitude=payload["latitude"],
        longitude=payload["longitude"],
        observed_at=payload.get("observed_at"),
        conditions=payload["features"],
        sources=payload["sources"],
    )


@app.get("/model/info")
async def model_info():
    """Get model information"""
    if model_instance is None or not model_instance.is_fitted:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    info = {
        "model_type": "Ensemble (Random Forest + XGBoost)",
        "version": "1.0.0",
        "features": model_instance.feature_names,
        "n_features": len(model_instance.feature_names) if model_instance.feature_names else 0,
        "training_history": model_instance.training_history,
        "risk_thresholds": {
            "LOW": f"0.00 - {config.risk_thresholds.low_max:.2f}",
            "MODERATE": f"{config.risk_thresholds.low_max:.2f} - {config.risk_thresholds.moderate_max:.2f}",
            "HIGH": f"{config.risk_thresholds.moderate_max:.2f} - 1.00"
        }
    }
    
    # Add feature importance if available
    try:
        importance = model_instance.get_feature_importance()
        info["feature_importance"] = importance.head(20).to_dict()
    except:
        pass
    
    return info


@app.get("/model/feature-importance")
async def feature_importance(top_n: int = 20):
    """Get feature importance"""
    if model_instance is None or not model_instance.is_fitted:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        importance = model_instance.get_feature_importance()
        return importance.head(top_n).to_dict()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/model/reload")
async def reload_model():
    """Reload model from disk"""
    success = load_model_and_preprocessors()
    if success:
        return {"status": "success", "message": "Model reloaded successfully"}
    else:
        raise HTTPException(status_code=500, detail="Failed to reload model")


# Error handlers
@app.exception_handler(ValueError)
async def value_error_handler(request: Request, exc: ValueError):
    return JSONResponse(
        status_code=400,
        content={"detail": str(exc)}
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error"}
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "src.api.main:app",
        host=config.api.host,
        port=config.api.port,
        reload=config.api.reload,
        workers=config.api.workers if not config.api.reload else 1
    )