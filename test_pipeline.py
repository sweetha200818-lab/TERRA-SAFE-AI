"""
Test script for TERRA SAFE AI pipeline
"""
import sys
from pathlib import Path
import pandas as pd
import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from src.utils.config import get_config
from src.utils.logger import setup_logger
from src.data.loader import DataLoader
from src.data.realtime import RealTimeDataFetcher
from src.data.features import create_pipeline
from src.models.model import create_model, EnsembleModel
from src.evaluation.evaluator import evaluate_model
from src.gis.processor import GISProcessor

setup_logger()

def test_realtime_data_fetch(n_samples: int = 60):
    """Fetch a labelled dataset from the live public sources"""
    print("\n" + "="*60)
    print("TEST 1: Real-Time Data Fetch")
    print("="*60)
    
    fetcher = RealTimeDataFetcher(get_config())
    df, coordinates = fetcher.build_training_dataset(n_samples=n_samples)
    print(f"Fetched {len(df)} records from live sources")
    print(f"Region: {fetcher.region_name} ({fetcher.bbox})")
    print(f"Columns: {list(df.columns)}")
    print(f"Landslide rate: {df['historical_landslide'].mean():.3f}")
    print(f"\nData:\n{df.head()}")
    print(f"\nStatistics:\n{df.describe()}")
    return df, coordinates, fetcher

def test_preprocessing(df):
    """Test data preprocessing"""
    print("\n" + "="*60)
    print("TEST 2: Data Preprocessing")
    print("="*60)
    
    config = get_config()
    loader = DataLoader(config)
    
    # Validate
    is_valid, errors = loader.validate_data(df)
    print(f"Data valid: {is_valid}")
    if errors:
        print(f"Errors: {errors}")
    
    # Preprocess
    df_processed = loader.preprocess_features(df, fit=True)
    print(f"Processed shape: {df_processed.shape}")
    print(f"Processed columns: {list(df_processed.columns)}")
    
    # Prepare target
    y = loader.prepare_target(df_processed)
    X = df_processed.drop(columns=[config.features.target[0]])
    print(f"Features: {X.shape}, Target: {y.shape}")
    print(f"Target distribution:\n{y.value_counts()}")
    
    # Feature engineering
    feature_pipe = create_pipeline()
    X_engineered = feature_pipe.fit_transform(X, y)
    print(f"Engineered features: {X_engineered.shape}")
    print(f"New columns: {list(X_engineered.columns)}")
    
    return X_engineered, y

def test_model_training(X, y):
    """Test model training"""
    print("\n" + "="*60)
    print("TEST 3: Model Training")
    print("="*60)
    
    config = get_config()
    loader = DataLoader(config)
    X_train, X_val, X_test, y_train, y_val, y_test = loader.split_data(X, y)
    print(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
    
    # Train Random Forest
    rf = create_model('random_forest', config)
    rf.fit(X_train, y_train, X_val, y_val)
    print(f"Random Forest trained - Train AUC: {rf.training_history.get('train_auc', 'N/A')}")
    
    # Train XGBoost
    xgb = create_model('xgboost', config)
    xgb.fit(X_train, y_train, X_val, y_val)
    print(f"XGBoost trained - Train AUC: {xgb.training_history.get('train_auc', 'N/A')}")
    
    # Train Ensemble
    ensemble = create_model('ensemble', config)
    ensemble.fit(X_train, y_train, X_val, y_val)
    print(f"Ensemble trained - Train AUC: {ensemble.training_history.get('train_auc', 'N/A')}")
    
    return ensemble, X_test, y_test

def test_evaluation(model, X_test, y_test):
    """Test model evaluation"""
    print("\n" + "="*60)
    print("TEST 4: Model Evaluation")
    print("="*60)
    
    results = evaluate_model(model, X_test, y_test, "test_ensemble")
    
    print(f"Accuracy:  {results['accuracy']:.4f}")
    print(f"Precision: {results['precision']:.4f}")
    print(f"Recall:    {results['recall']:.4f}")
    print(f"F1-Score:  {results['f1_score']:.4f}")
    print(f"ROC-AUC:   {results['roc_auc']:.4f}")
    print(f"PR-AUC:    {results['pr_auc']:.4f}")
    print(f"Brier:     {results['brier_score']:.4f}")
    print(f"Optimal Threshold: {results['optimal_threshold']:.4f}")
    print(f"\nConfusion Matrix:")
    cm = np.array(results['confusion_matrix'])
    print(f"  TN={cm[0,0]}, FP={cm[0,1]}")
    print(f"  FN={cm[1,0]}, TP={cm[1,1]}")
    
    return results

def test_gis_processing(model, X_test, y_test, coordinates, fetcher):
    """Test GIS processing"""
    print("\n" + "="*60)
    print("TEST 5: GIS Risk Maps")
    print("="*60)
    
    config = get_config()
    gis = GISProcessor(config)
    
    # Get predictions
    y_prob = model.predict_proba(X_test)
    y_pred = model.predict(X_test)
    
    # Real sample locations from the live data sources
    test_coords = coordinates.loc[X_test.index]
    n_samples = len(X_test)
    print(f"Using {n_samples} real coordinates from {fetcher.region_name}")
    
    import geopandas as gpd
    from shapely.geometry import Point
    geometry = [Point(lon, lat) for lat, lon
                in zip(test_coords['latitude'], test_coords['longitude'])]
    gdf = gpd.GeoDataFrame(
        {'probability': y_prob, 'predicted': y_pred, 'actual': y_test.values},
        geometry=geometry,
        crs=config.gis.crs
    )
    
    output_dir = Path("outputs/test_risk_layers")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create susceptibility map
    sus_map = gis.create_susceptibility_map(gdf, 'probability', output_dir / 'susceptibility.geojson')
    print(f"Susceptibility map: {len(sus_map)} features")
    
    # Create heatmap
    heatmap = gis.create_risk_heatmap(gdf, 'probability', output_dir / 'heatmap.geojson')
    print(f"Risk heatmap: {len(heatmap)} grid cells")
    
    # High risk zones
    zones = gis.identify_high_risk_zones(gdf, 'probability', output_path=output_dir / 'zones.geojson')
    print(f"High-risk zones: {len(zones)} zones")
    
    # Real administrative boundaries and roads from OpenStreetMap
    layers = fetcher.region_layers()
    admin = layers['admin']
    roads = layers['roads']
    
    if not admin.empty:
        district_risk = gis.district_level_risk(gdf, 'probability', admin, output_path=output_dir / 'district.geojson')
        print(f"District risk: {len(district_risk)} districts")
    else:
        print("No administrative boundaries returned by OpenStreetMap - skipped")
    
    if not roads.empty:
        road_risk = gis.road_risk_analysis(gdf, 'probability', roads, output_path=output_dir / 'road_risk.csv')
        print(f"Road risk analysis: {len(road_risk)} buffer distances")
        
        routes = gis.safe_unsafe_routes(gdf, 'probability', roads, output_path=output_dir / 'routes.geojson')
        safe_count = (routes['classification'] == 'SAFE').sum()
        unsafe_count = (routes['classification'] == 'UNSAFE').sum()
        print(f"Routes: {safe_count} safe, {unsafe_count} unsafe")
    else:
        print("No roads returned by OpenStreetMap - skipped")
    
    return True

def test_api_prediction():
    """Test API prediction logic"""
    print("\n" + "="*60)
    print("TEST 6: API Prediction Logic")
    print("="*60)
    
    # This would require a trained model saved to disk
    # For now, just verify the input validation works
    from src.api.main import PredictionInput, BatchPredictionInput
    
    # Single prediction
    single = PredictionInput(
        rainfall_mm=180,
        soil_moisture=0.42,
        temperature_c=25,
        humidity_percent=70,
        slope_degrees=38,
        elevation_m=1450,
        distance_to_road_km=2.5,
        land_cover="forest"
    )
    print(f"Single input valid: {single}")
    
    # Batch prediction
    batch = BatchPredictionInput(predictions=[single, single])
    print(f"Batch input valid: {len(batch.predictions)} predictions")
    
    return True

def main():
    """Run all tests"""
    print("="*60)
    print("TERRA SAFE AI - PIPELINE TESTS")
    print("="*60)
    
    try:
        # Test 1: real-time data fetch
        df, coordinates, fetcher = test_realtime_data_fetch()
        
        # Test 2: Preprocessing
        X, y = test_preprocessing(df)
        
        # Test 3: Model training
        model, X_test, y_test = test_model_training(X, y)
        
        # Test 4: Evaluation
        results = test_evaluation(model, X_test, y_test)
        
        # Test 5: GIS processing
        test_gis_processing(model, X_test, y_test, coordinates, fetcher)
        
        # Test 6: API validation
        test_api_prediction()
        
        print("\n" + "="*60)
        print("ALL TESTS PASSED!")
        print("="*60)
        return True
        
    except Exception as e:
        print(f"\nTEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)