"""
Main training pipeline for TERRA SAFE AI
"""
import argparse
import sys
from pathlib import Path
from typing import Optional, Dict, Any
import pandas as pd
import geopandas as gpd

from src.utils.config import get_config, load_config
from src.utils.logger import setup_logger, get_logger
from src.data.loader import DataLoader
from src.data.realtime import RealTimeDataFetcher
from src.data.features import create_pipeline
from src.models.model import create_model, EnsembleModel
from src.evaluation.evaluator import ModelEvaluator, evaluate_model
from src.gis.processor import GISProcessor
from shapely.geometry import Point

logger = get_logger(__name__)


class TrainingPipeline:
    """Complete training pipeline for landslide prediction"""
    
    def __init__(self, config_path: Optional[str] = None):
        self.config = load_config(config_path) if config_path else get_config()
        self.data_loader = DataLoader(self.config)
        self.feature_pipeline = create_pipeline()
        self.models = {}
        self.evaluator = ModelEvaluator(self.config)
        self.gis_processor = GISProcessor(self.config)
        self.fetcher = RealTimeDataFetcher(self.config)
        self.coordinates: Optional[pd.DataFrame] = None
        self.results = {}
    
    def load_data(self, data_path: Optional[str] = None,
                  n_samples: Optional[int] = None,
                  refresh: bool = False) -> pd.DataFrame:
        """Load a CSV of records or fetch a real-time dataset."""
        if data_path:
            logger.info(f"Loading data from {data_path}")
            df = self.data_loader.load_csv(data_path)
            if {"latitude", "longitude"}.issubset(df.columns):
                coord_cols = [c for c in ("latitude", "longitude", "event_date")
                              if c in df.columns]
                self.coordinates = df[coord_cols].copy()
                self.coordinates.index = df.index
                df = df.drop(columns=coord_cols)
            else:
                logger.warning("CSV has no latitude/longitude columns - "
                               "GIS risk maps will be skipped")
                self.coordinates = None
        else:
            self.fetcher.refresh = refresh
            logger.info("Fetching real-time training data (live public sources)")
            df, self.coordinates = self.fetcher.build_training_dataset(n_samples=n_samples)

            processed_dir = Path(self.config.get("paths.processed_data", "data/processed"))
            processed_dir.mkdir(parents=True, exist_ok=True)
            export = df.copy()
            for column in self.coordinates.columns:
                export[column] = self.coordinates[column].values
            out_path = processed_dir / "realtime_training_data.csv"
            export.to_csv(out_path, index=False)
            logger.info(f"Real-time dataset saved to {out_path}")
        
        # Validate
        is_valid, errors = self.data_loader.validate_data(df)
        if not is_valid:
            raise ValueError(f"Data validation failed: {errors}")
        
        return df
    
    def prepare_data(self, df: pd.DataFrame) -> tuple:
        """Prepare features and target, split data"""
        # Preprocess features
        df_processed = self.data_loader.preprocess_features(df, fit=True)
        
        # Prepare target
        y = self.data_loader.prepare_target(df_processed)
        X = df_processed.drop(columns=[self.config.features.target[0]])
        
        # Feature engineering
        logger.info("Applying feature engineering...")
        X_engineered = self.feature_pipeline.fit_transform(X, y)
        
        # Split data
        logger.info("Splitting data...")
        X_train, X_val, X_test, y_train, y_val, y_test = self.data_loader.split_data(
            X_engineered, y
        )
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def train_models(self, X_train: pd.DataFrame, y_train: pd.Series,
                     X_val: pd.DataFrame, y_val: pd.Series,
                     model_types: Optional[list] = None) -> Dict[str, Any]:
        """Train specified models"""
        model_types = model_types or ['random_forest', 'xgboost', 'ensemble']
        
        for model_type in model_types:
            logger.info(f"Training {model_type}...")
            model = create_model(model_type, self.config)
            model.fit(X_train, y_train, X_val, y_val)
            self.models[model_type] = model
        
        return self.models
    
    def evaluate_models(self, X_test: pd.DataFrame, y_test: pd.Series,
                        output_dir: Optional[str] = None) -> Dict[str, Any]:
        """Evaluate all trained models"""
        for name, model in self.models.items():
            logger.info(f"Evaluating {name}...")
            results = evaluate_model(model, X_test, y_test, name, output_dir)
            self.results[name] = results
        
        # Compare models
        comparison = self.evaluator.compare_models()
        logger.info(f"\nModel Comparison:\n{comparison.to_string(index=False)}")
        
        if output_dir:
            comparison.to_csv(Path(output_dir) / "model_comparison.csv", index=False)
            self.evaluator.save_results(Path(output_dir) / "evaluation_results.json")
        
        return self.results
    
    def select_best_model(self, metric: str = 'roc_auc') -> tuple:
        """Select best model based on metric"""
        best_score = -1
        best_name = None
        best_model = None
        
        for name, results in self.results.items():
            score = results.get(metric, 0)
            if score > best_score:
                best_score = score
                best_name = name
                best_model = self.models[name]
        
        logger.info(f"Best model: {best_name} ({metric}={best_score:.4f})")
        return best_name, best_model
    
    def save_artifacts(self, output_dir: str, best_model_name: Optional[str] = None):
        """Save model artifacts and preprocessors"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save preprocessors
        self.data_loader.save_preprocessors(output_path)
        
        # Save feature pipeline
        import joblib
        joblib.dump(self.feature_pipeline, output_path / "feature_pipeline.joblib")
        
        # Save models
        for name, model in self.models.items():
            model.save(output_path / name)
        
        # Save best model separately if specified
        if best_model_name and best_model_name in self.models:
            best_model = self.models[best_model_name]
            best_model.save(output_path / "best_model")
            logger.info(f"Best model ({best_model_name}) saved as 'best_model'")
        
        # Save config
        import yaml
        with open(output_path / "config.yaml", 'w') as f:
            yaml.dump(self.config.to_dict(), f)
        
        logger.info(f"All artifacts saved to {output_path}")
    
    def create_risk_maps(self, X_test: pd.DataFrame, y_test: pd.Series,
                         best_model_name: str,
                         output_dir: str,
                         admin_boundaries: Optional[gpd.GeoDataFrame] = None,
                         road_network: Optional[gpd.GeoDataFrame] = None,
                         coordinates: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
        """Create GIS risk layers from test predictions using real coordinates"""
        best_model = self.models[best_model_name]
        
        coordinates = coordinates if coordinates is not None else self.coordinates
        if coordinates is None:
            raise ValueError(
                "Real coordinates are required to build risk maps. "
                "Fetch real-time data or provide latitude/longitude columns."
            )
        coordinates = coordinates.loc[X_test.index]
        
        # Get predictions on test set
        y_prob = best_model.predict_proba(X_test)
        y_pred = best_model.predict(X_test)
        
        # Real sample locations (latitude/longitude from the live data sources)
        geometry = [Point(lon, lat) for lat, lon
                    in zip(coordinates["latitude"], coordinates["longitude"])]
        gdf = gpd.GeoDataFrame(
            {'probability': y_prob, 'predicted': y_pred, 'actual': y_test.values},
            geometry=geometry,
            crs=self.config.gis.crs
        )
        
        output_path = Path(output_dir) / "risk_layers"
        output_path.mkdir(parents=True, exist_ok=True)
        
        layers = {}
        
        # 1. Susceptibility map
        layers['susceptibility'] = self.gis_processor.create_susceptibility_map(
            gdf, 'probability', output_path / 'susceptibility_map.geojson'
        )
        
        # 2. Risk heatmap
        layers['heatmap'] = self.gis_processor.create_risk_heatmap(
            gdf, 'probability', output_path / 'risk_heatmap.geojson'
        )
        
        # 3. High-risk zones
        layers['high_risk_zones'] = self.gis_processor.identify_high_risk_zones(
            gdf, 'probability', output_path=output_path / 'high_risk_zones.geojson'
        )
        
        # 4. District-level risk (real administrative boundaries)
        if admin_boundaries is not None and not admin_boundaries.empty:
            layers['district_risk'] = self.gis_processor.district_level_risk(
                gdf, 'probability', admin_boundaries, output_path=output_path / 'district_risk.geojson'
            )
        else:
            logger.warning("No administrative boundaries available - skipping district risk layer")
        
        # 5. Road risk (real road network)
        if road_network is not None and not road_network.empty:
            layers['road_risk'] = self.gis_processor.road_risk_analysis(
                gdf, 'probability', road_network, output_path=output_path / 'road_risk.csv'
            )
            layers['routes'] = self.gis_processor.safe_unsafe_routes(
                gdf, 'probability', road_network, output_path=output_path / 'routes.geojson'
            )
        else:
            logger.warning("No road network available - skipping road risk layers")
        
        logger.info(f"Risk maps created in {output_path}")
        return layers
    
    def run_full_pipeline(self, 
                          data_path: Optional[str] = None,
                          n_samples: Optional[int] = None,
                          model_types: Optional[list] = None,
                          output_dir: str = "models",
                          create_maps: bool = True,
                          refresh: bool = False) -> Dict[str, Any]:
        """Run the complete training pipeline"""
        logger.info("=" * 60)
        logger.info("STARTING TERRA SAFE AI TRAINING PIPELINE")
        logger.info("=" * 60)
        
        # 1. Load data (CSV or real-time fetch)
        df = self.load_data(data_path, n_samples=n_samples, refresh=refresh)
        
        # 2. Prepare data
        X_train, X_val, X_test, y_train, y_val, y_test = self.prepare_data(df)
        
        # 3. Train models
        self.train_models(X_train, y_train, X_val, y_val, model_types)
        
        # 4. Evaluate models
        self.evaluate_models(X_test, y_test, output_dir)
        
        # 5. Select best model
        best_name, best_model = self.select_best_model()
        
        # 6. Save artifacts
        self.save_artifacts(output_dir, best_name)
        
        # 7. Create risk maps (optional)
        if create_maps:
            if self.coordinates is None:
                logger.warning("No real coordinates available - skipping GIS risk map creation")
            else:
                # Real administrative boundaries and roads from OpenStreetMap
                region_layers = self.fetcher.region_layers()
                self.create_risk_maps(
                    X_test, y_test, best_name, output_dir,
                    admin_boundaries=region_layers['admin'],
                    road_network=region_layers['roads'],
                    coordinates=self.coordinates,
                )
        
        logger.info("=" * 60)
        logger.info("TRAINING PIPELINE COMPLETED SUCCESSFULLY")
        logger.info("=" * 60)
        
        return {
            'best_model': best_name,
            'best_model_object': best_model,
            'results': self.results,
            'models': self.models
        }


def main():
    """CLI entry point"""
    parser = argparse.ArgumentParser(description="TERRA SAFE AI Training Pipeline")
    parser.add_argument("--data", type=str, help="Path to training data CSV")
    parser.add_argument("--n-samples", type=int, default=None,
                        help="Number of real-time records to fetch "
                             "(default: realtime.n_samples from config.yaml)")
    parser.add_argument("--refresh", action="store_true",
                        help="Ignore the local cache and re-download live data")
    parser.add_argument("--models", nargs="+", default=["random_forest", "xgboost", "ensemble"],
                        choices=["random_forest", "xgboost", "ensemble"],
                        help="Models to train")
    parser.add_argument("--output", type=str, default="models", help="Output directory")
    parser.add_argument("--config", type=str, help="Path to config file")
    parser.add_argument("--no-maps", action="store_true", help="Skip GIS risk map creation")
    parser.add_argument("--eval-only", type=str, help="Evaluate existing model (provide model path)")
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logger()
    
    # Load config
    if args.config:
        load_config.cache_clear()
        config = load_config(args.config)
    else:
        config = get_config()
    
    # Create pipeline
    pipeline = TrainingPipeline(args.config)
    
    if args.eval_only:
        # Evaluation mode
        logger.info(f"Loading model from {args.eval_only}")
        model = EnsembleModel(config)
        model.load(args.eval_only)
        
        # Load test data (CSV or real-time fetch)
        if args.data:
            df = pipeline.load_data(args.data)
        else:
            df = pipeline.load_data(n_samples=args.n_samples, refresh=args.refresh)
        
        X_train, X_val, X_test, y_train, y_val, y_test = pipeline.prepare_data(df)
        
        results = evaluate_model(model, X_test, y_test, "loaded_model", args.output)
        print(f"\nEvaluation Results:")
        print(f"  Accuracy: {results['accuracy']:.4f}")
        print(f"  Precision: {results['precision']:.4f}")
        print(f"  Recall: {results['recall']:.4f}")
        print(f"  F1-Score: {results['f1_score']:.4f}")
        print(f"  ROC-AUC: {results['roc_auc']:.4f}")
    else:
        # Training mode
        pipeline.run_full_pipeline(
            data_path=args.data,
            n_samples=args.n_samples,
            model_types=args.models,
            output_dir=args.output,
            create_maps=not args.no_maps,
            refresh=args.refresh
        )


if __name__ == "__main__":
    main()