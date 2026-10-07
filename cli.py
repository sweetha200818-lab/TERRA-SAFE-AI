#!/usr/bin/env python
"""
TERRA SAFE AI - Command Line Interface
"""
import argparse
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from src.pipeline import TrainingPipeline
from src.utils.config import get_config, load_config
from src.utils.logger import setup_logger, get_logger
from src.api.main import app
import uvicorn

logger = get_logger(__name__)


def cmd_train(args):
    """Train models"""
    setup_logger()
    
    if args.config:
        load_config.cache_clear()
        config = load_config(args.config)
    else:
        config = get_config()
    
    pipeline = TrainingPipeline(args.config)
    
    pipeline.run_full_pipeline(
        data_path=args.data,
        n_samples=args.n_samples,
        model_types=args.models,
        output_dir=args.output,
        create_maps=not args.no_maps,
        refresh=args.refresh
    )


def cmd_evaluate(args):
    """Evaluate a trained model"""
    setup_logger()
    
    if args.config:
        load_config.cache_clear()
        config = load_config(args.config)
    else:
        config = get_config()
    
    from src.models.model import EnsembleModel
    from src.evaluation.evaluator import evaluate_model
    from src.data.loader import DataLoader
    
    # Load model
    model = EnsembleModel(config)
    model.load(args.model_path)
    
    # Load data (CSV or real-time fetch)
    pipeline = TrainingPipeline(args.config)
    if args.data:
        df = pipeline.load_data(args.data)
    else:
        df = pipeline.load_data(n_samples=args.n_samples, refresh=args.refresh)
    
    X_train, X_val, X_test, y_train, y_val, y_test = pipeline.prepare_data(df)
    
    results = evaluate_model(model, X_test, y_test, "evaluated_model", args.output)
    
    print(f"\nEvaluation Results:")
    print(f"  Accuracy:  {results['accuracy']:.4f}")
    print(f"  Precision: {results['precision']:.4f}")
    print(f"  Recall:    {results['recall']:.4f}")
    print(f"  F1-Score:  {results['f1_score']:.4f}")
    print(f"  ROC-AUC:   {results['roc_auc']:.4f}")
    print(f"  PR-AUC:    {results['pr_auc']:.4f}")


def cmd_predict(args):
    """Make predictions using trained model"""
    setup_logger()
    
    if args.config:
        load_config.cache_clear()
        config = load_config(args.config)
    else:
        config = get_config()
    
    from src.models.model import EnsembleModel
    from src.data.loader import DataLoader
    from src.data.realtime import RealTimeDataFetcher
    from pathlib import Path
    
    model_path = Path(args.model_path)
    # Determine preprocessor path (parent if in ensemble/rf/xgb subdir)
    preproc_path = model_path if (model_path / "scaler.joblib").exists() else model_path.parent
    
    # Load model
    model = EnsembleModel(config)
    model.load(str(model_path))
    
    # Load preprocessors
    data_loader = DataLoader(config)
    data_loader.load_preprocessors(str(preproc_path))
    
    # Prepare input: live conditions for a coordinate, a CSV, or explicit values
    if args.latitude is not None and args.longitude is not None:
        fetcher = RealTimeDataFetcher(config, refresh=args.refresh)
        live = fetcher.fetch_live_conditions(args.latitude, args.longitude)
        df = pd.DataFrame([live["features"]])
        print(f"\nLive conditions for ({args.latitude:.5f}, {args.longitude:.5f}) "
              f"observed at {live['observed_at']}:")
        for key, value in live["features"].items():
            print(f"  {key:22s}: {value}")
        print(f"  sources               : {live['sources']}")
    elif args.input_file:
        df = pd.read_csv(args.input_file)
    else:
        # Single prediction from args
        df = pd.DataFrame([{
            'rainfall_mm': args.rainfall,
            'soil_moisture': args.soil_moisture,
            'temperature_c': args.temperature,
            'humidity_percent': args.humidity,
            'slope_degrees': args.slope,
            'elevation_m': args.elevation,
            'distance_to_road_km': args.distance_to_road,
            'land_cover': args.land_cover
        }])
    
    # Impute anything the live sources could not return, then preprocess
    processed = data_loader.preprocess_features(data_loader.fill_missing_from_scaler(df), fit=False)
    
    # Ensure all features present
    feature_names = model.feature_names
    if feature_names:
        for feat in feature_names:
            if feat not in processed.columns:
                processed[feat] = 0
        processed = processed[feature_names]
    
    # Predict
    probabilities = model.predict_proba(processed)
    predictions = model.predict(processed)
    
    # Output results
    for i, (prob, pred) in enumerate(zip(probabilities, predictions)):
        if prob <= config.risk_thresholds.low_max:
            risk = "LOW"
        elif prob <= config.risk_thresholds.moderate_max:
            risk = "MODERATE"
        else:
            risk = "HIGH"
        
        print(f"\nPrediction {i+1}:")
        print(f"  Probability: {prob:.4f}")
        print(f"  Risk Level:  {risk}")
        print(f"  Risk Score:  {prob*100:.1f}/100")
        print(f"  Class:       {'Landslide' if pred == 1 else 'No Landslide'}")


def cmd_serve(args):
    """Start API server"""
    setup_logger()
    
    if args.config:
        load_config.cache_clear()
        config = load_config(args.config)
    else:
        config = get_config()
    
    # Override config with CLI args
    config.api.host = args.host
    config.api.port = args.port
    config.api.reload = args.reload
    
    uvicorn.run(
        "src.api.main:app",
        host=config.api.host,
        port=config.api.port,
        reload=config.api.reload,
        workers=config.api.workers if not config.api.reload else 1
    )


def cmd_fetch_data(args):
    """Download a real-time training dataset from the live public sources"""
    setup_logger()
    
    from src.data.realtime import RealTimeDataFetcher
    
    fetcher = RealTimeDataFetcher(config=load_config(args.config) if args.config else get_config(),
                                  refresh=args.refresh)
    features, coordinates = fetcher.build_training_dataset(n_samples=args.n_samples)
    
    dataset = features.copy()
    for column in coordinates.columns:
        dataset[column] = coordinates[column].values
    
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_csv(output_path, index=False)
    
    print(f"Fetched {len(dataset)} real-time records for {fetcher.region_name}")
    print(f"Saved to {output_path}")
    print(f"Landslide rate: {features['historical_landslide'].mean():.3f}")
    print(f"Sources: NASA COOLR catalog, Open-Meteo, OpenStreetMap")
    print(f"\nFeature statistics:")
    print(features.describe())


def main():
    parser = argparse.ArgumentParser(
        description="TERRA SAFE AI - Landslide Risk Prediction System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train with real-time data fetched from live public sources
  python cli.py train
  
  # Train with a custom dataset size
  python cli.py train --n-samples 240 --output models/v1
  
  # Train with your own CSV
  python cli.py train --data data/training.csv --output models/v1
  
  # Evaluate existing model (uses real-time data when --data is omitted)
  python cli.py evaluate --model-path models/ensemble --data data/test.csv
  
  # Predict from live conditions at a coordinate
  python cli.py predict --model-path models/ensemble --latitude 30.06 --longitude 79.02
  
  # Predict from explicit feature values
  python cli.py predict --model-path models/ensemble --rainfall 180 --soil-moisture 0.42 --slope 38 --elevation 1450 --land-cover forest
  
  # Start API server
  python cli.py serve --host 0.0.0.0 --port 8000
  
  # Download real-time training data to CSV
  python cli.py fetch-data --n-samples 180 --output data/training.csv
        """
    )
    
    parser.add_argument("--config", type=str, help="Path to config file")
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Train command
    train_parser = subparsers.add_parser("train", help="Train models")
    train_parser.add_argument("--data", type=str, help="Path to training data CSV")
    train_parser.add_argument("--n-samples", type=int, default=None,
                              help="Number of real-time records to fetch "
                                   "(default: realtime.n_samples from config.yaml)")
    train_parser.add_argument("--refresh", action="store_true",
                              help="Ignore local cache and re-download live data")
    train_parser.add_argument("--models", nargs="+", default=["random_forest", "xgboost", "ensemble"],
                              choices=["random_forest", "xgboost", "ensemble"])
    train_parser.add_argument("--output", type=str, default="models", help="Output directory")
    train_parser.add_argument("--no-maps", action="store_true", help="Skip GIS risk map creation")
    train_parser.set_defaults(func=cmd_train)
    
    # Evaluate command
    eval_parser = subparsers.add_parser("evaluate", help="Evaluate trained model")
    eval_parser.add_argument("--model-path", type=str, required=True, help="Path to model directory")
    eval_parser.add_argument("--data", type=str, help="Path to test data CSV")
    eval_parser.add_argument("--n-samples", type=int, default=None,
                             help="Number of real-time records to fetch if --data is omitted")
    eval_parser.add_argument("--refresh", action="store_true",
                             help="Ignore local cache and re-download live data")
    eval_parser.add_argument("--output", type=str, help="Output directory for plots")
    eval_parser.set_defaults(func=cmd_evaluate)
    
    # Predict command
    pred_parser = subparsers.add_parser("predict", help="Make predictions")
    pred_parser.add_argument("--model-path", type=str, required=True, help="Path to model directory")
    pred_parser.add_argument("--input-file", type=str, help="CSV file with input data")
    pred_parser.add_argument("--latitude", type=float, default=None,
                             help="Latitude - fetches live conditions for this coordinate")
    pred_parser.add_argument("--longitude", type=float, default=None,
                             help="Longitude - fetches live conditions for this coordinate")
    pred_parser.add_argument("--refresh", action="store_true",
                             help="Ignore local cache and re-download live data")
    pred_parser.add_argument("--rainfall", type=float, default=180, help="Rainfall (mm)")
    pred_parser.add_argument("--soil-moisture", type=float, default=0.42, help="Soil moisture (0-1)")
    pred_parser.add_argument("--temperature", type=float, default=25, help="Temperature (C)")
    pred_parser.add_argument("--humidity", type=float, default=70, help="Humidity (%)")
    pred_parser.add_argument("--slope", type=float, default=38, help="Slope (degrees)")
    pred_parser.add_argument("--elevation", type=float, default=1450, help="Elevation (m)")
    pred_parser.add_argument("--distance-to-road", type=float, default=2.5, help="Distance to road (km)")
    pred_parser.add_argument("--land-cover", type=str, default="forest", help="Land cover type")
    pred_parser.set_defaults(func=cmd_predict)
    
    # Serve command
    serve_parser = subparsers.add_parser("serve", help="Start API server")
    serve_parser.add_argument("--host", type=str, default="0.0.0.0", help="Host address")
    serve_parser.add_argument("--port", type=int, default=8000, help="Port number")
    serve_parser.add_argument("--reload", action="store_true", help="Enable auto-reload")
    serve_parser.set_defaults(func=cmd_serve)
    
    # Fetch data command
    gen_parser = subparsers.add_parser("fetch-data",
                                       help="Download real-time training data from live sources")
    gen_parser.add_argument("--n-samples", type=int, default=None,
                            help="Number of real-time records to fetch "
                                 "(default: realtime.n_samples from config.yaml)")
    gen_parser.add_argument("--refresh", action="store_true",
                            help="Ignore local cache and re-download live data")
    gen_parser.add_argument("--output", type=str, default="data/realtime_training_data.csv",
                            help="Output file")
    gen_parser.set_defaults(func=cmd_fetch_data)
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        sys.exit(1)
    
    # Import pandas here to avoid import issues
    import pandas as pd
    globals()['pd'] = pd
    
    args.func(args)


if __name__ == "__main__":
    main()