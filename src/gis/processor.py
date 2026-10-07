"""
GIS processing for TERRA SAFE AI - Risk maps, heatmaps, and spatial analysis
"""
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.transform import from_bounds
from rasterio.features import rasterize
from rasterio.warp import reproject, Resampling
import rasterio.mask
from shapely.geometry import Point, Polygon, box, mapping
from shapely.ops import unary_union
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union
import warnings
warnings.filterwarnings('ignore', category=UserWarning, module='rasterio')

from src.utils.config import get_config
from src.utils.logger import get_logger

logger = get_logger(__name__)


class GISProcessor:
    """Main GIS processing class for creating risk maps and spatial analysis"""
    
    def __init__(self, config: Optional[Any] = None):
        self.config = config or get_config()
        self.crs = self.config.gis.crs
        self.target_crs = self.config.gis.target_crs
        self.resolution = self.config.gis.raster_resolution
    
    def create_risk_raster(self, 
                          predictions: np.ndarray,
                          coordinates: np.ndarray,
                          bounds: Optional[Tuple[float, float, float, float]] = None,
                          output_path: Optional[str] = None) -> rasterio.DatasetReader:
        """
        Create a risk raster from point predictions
        
        Args:
            predictions: Array of risk probabilities (0-1)
            coordinates: Array of (lon, lat) coordinates
            bounds: Optional (minx, miny, maxx, maxy) bounds
            output_path: Optional path to save GeoTIFF
            
        Returns:
            Rasterio dataset reader
        """
        if bounds is None:
            minx, miny = coordinates.min(axis=0)
            maxx, maxy = coordinates.max(axis=0)
            bounds = (minx, miny, maxx, maxy)
        
        # Calculate raster dimensions
        width = int((bounds[2] - bounds[0]) / (self.resolution / 111000))  # approx degrees to meters
        height = int((bounds[3] - bounds[1]) / (self.resolution / 111000))
        width = max(width, 100)
        height = max(height, 100)
        
        # Create transform
        transform = from_bounds(*bounds, width, height)
        
        # Create empty raster
        raster = np.full((height, width), np.nan, dtype=np.float32)
        
        # Rasterize points
        shapes = [(Point(xy), val) for xy, val in zip(coordinates, predictions)]
        rasterized = rasterize(
            shapes,
            out_shape=(height, width),
            transform=transform,
            fill=np.nan,
            dtype=np.float32
        )
        
        # Interpolate missing values using nearest neighbor
        from scipy.ndimage import gaussian_filter
        mask = np.isnan(rasterized)
        if mask.any() and not mask.all():
            # Fill with nearest neighbor interpolation
            from scipy.ndimage import distance_transform_edt
            indices = distance_transform_edt(mask, return_indices=True)[1]
            rasterized[mask] = rasterized[tuple(indices[:, mask])]
        
        # Smooth the raster
        rasterized = gaussian_filter(rasterized, sigma=1)
        
        if output_path:
            self._save_raster(rasterized, transform, output_path)
        
        return rasterized, transform
    
    def _save_raster(self, data: np.ndarray, transform: rasterio.Affine, 
                     output_path: str, crs: Optional[str] = None):
        """Save raster to GeoTIFF"""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        crs = crs or self.crs
        profile = {
            'driver': 'GTiff',
            'dtype': 'float32',
            'width': data.shape[1],
            'height': data.shape[0],
            'count': 1,
            'crs': crs,
            'transform': transform,
            'compress': self.config.gis.compression,
            'nodata': -9999
        }
        
        with rasterio.open(output_path, 'w', **profile) as dst:
            dst.write(data.astype(np.float32), 1)
        
        logger.info(f"Raster saved to {output_path}")
    
    def create_susceptibility_map(self, 
                                 gdf: gpd.GeoDataFrame,
                                 probability_column: str,
                                 output_path: str,
                                 classification: str = 'quantile',
                                 n_classes: int = 5) -> gpd.GeoDataFrame:
        """
        Create landslide susceptibility map from GeoDataFrame with probabilities
        """
        gdf = gdf.copy()
        
        # Classify probabilities
        if classification == 'quantile':
            gdf['susceptibility_class'] = pd.qcut(
                gdf[probability_column], 
                q=n_classes, 
                labels=[f'Class {i+1}' for i in range(n_classes)],
                duplicates='drop'
            )
        elif classification == 'equal_interval':
            gdf['susceptibility_class'] = pd.cut(
                gdf[probability_column],
                bins=n_classes,
                labels=[f'Class {i+1}' for i in range(n_classes)]
            )
        elif classification == 'risk_thresholds':
            thresholds = self.config.risk_thresholds
            gdf['susceptibility_class'] = pd.cut(
                gdf[probability_column],
                bins=[0, thresholds.low_max, thresholds.moderate_max, 1.0],
                labels=['LOW', 'MODERATE', 'HIGH']
            )
        
        # Save
        gdf.to_file(output_path, driver='GeoJSON')
        logger.info(f"Susceptibility map saved to {output_path}")
        
        return gdf
    
    def create_risk_heatmap(self,
                           gdf: gpd.GeoDataFrame,
                           probability_column: str,
                           output_path: str,
                           grid_size: float = 0.01) -> gpd.GeoDataFrame:
        """
        Create risk heatmap using hexagonal/grid aggregation
        """
        # Create grid
        bounds = gdf.total_bounds
        x_min, y_min, x_max, y_max = bounds
        
        # Create fishnet grid
        x_coords = np.arange(x_min, x_max, grid_size)
        y_coords = np.arange(y_min, y_max, grid_size)
        
        polygons = []
        for x in x_coords:
            for y in y_coords:
                polygons.append(box(x, y, x + grid_size, y + grid_size))
        
        grid = gpd.GeoDataFrame(geometry=polygons, crs=gdf.crs)
        
        # Spatial join and aggregate
        # Reset index on grid to ensure we can group by it
        grid_reset = grid.reset_index().rename(columns={'index': 'grid_id'})
        joined = gpd.sjoin(grid_reset, gdf[[probability_column, 'geometry']], how='left', predicate='intersects')
        
        # Group by grid_id and aggregate, keeping only grid geometry
        heatmap = joined.groupby('grid_id').agg({
            probability_column: ['mean', 'max', 'count'],
            'geometry': 'first'  # This is the grid geometry
        }).reset_index()
        
        heatmap.columns = ['grid_id', 'mean_prob', 'max_prob', 'count', 'geometry']
        heatmap = gpd.GeoDataFrame(heatmap, geometry='geometry', crs=gdf.crs)
        heatmap = heatmap.dropna(subset=['mean_prob'])
        
        # Classify heatmap
        heatmap['risk_level'] = pd.cut(
            heatmap['mean_prob'],
            bins=[0, self.config.risk_thresholds.low_max, 
                  self.config.risk_thresholds.moderate_max, 1.0],
            labels=['LOW', 'MODERATE', 'HIGH']
        )
        
        heatmap.to_file(output_path, driver='GeoJSON')
        logger.info(f"Risk heatmap saved to {output_path}")
        
        return heatmap
    
    def identify_high_risk_zones(self,
                                gdf: gpd.GeoDataFrame,
                                probability_column: str,
                                threshold: float = 0.6,
                                min_area_km2: float = 0.1,
                                output_path: Optional[str] = None) -> gpd.GeoDataFrame:
        """
        Identify contiguous high-risk zones
        """
        # Filter high risk
        high_risk = gdf[gdf[probability_column] >= threshold].copy()
        
        if high_risk.empty:
            logger.warning("No high-risk areas found")
            return gpd.GeoDataFrame(columns=['geometry'], crs=gdf.crs)
        
        # Dissolve contiguous areas
        dissolved = high_risk.dissolve()
        
        # Explode multipart geometries
        exploded = dissolved.explode(index_parts=True).reset_index(drop=True)
        
        # Calculate area in km2
        exploded_projected = exploded.to_crs(self.target_crs)
        exploded['area_km2'] = exploded_projected.geometry.area / 1e6
        
        # Filter by minimum area
        zones = exploded[exploded['area_km2'] >= min_area_km2].copy()
        zones = zones.to_crs(self.crs)
        
        # Add zone ID and statistics
        zones['zone_id'] = range(1, len(zones) + 1)
        
        # Calculate stats for each zone
        for idx, zone in zones.iterrows():
            contained = gdf[gdf.within(zone.geometry)]
            zones.loc[idx, 'max_prob'] = contained[probability_column].max()
            zones.loc[idx, 'mean_prob'] = contained[probability_column].mean()
            zones.loc[idx, 'point_count'] = len(contained)
        
        if output_path:
            zones.to_file(output_path, driver='GeoJSON')
            logger.info(f"High-risk zones saved to {output_path}")
        
        return zones
    
    def district_level_risk(self,
                           gdf: gpd.GeoDataFrame,
                           probability_column: str,
                           admin_boundaries: gpd.GeoDataFrame,
                           admin_id_column: str = 'district_id',
                           admin_name_column: str = 'district_name',
                           output_path: Optional[str] = None) -> gpd.GeoDataFrame:
        """
        Aggregate risk to district/administrative level
        """
        # Ensure same CRS
        if gdf.crs != admin_boundaries.crs:
            gdf = gdf.to_crs(admin_boundaries.crs)
        
        # Spatial join
        joined = gpd.sjoin(gdf, admin_boundaries[[admin_id_column, admin_name_column, 'geometry']], 
                          how='left', predicate='within')
        
        # Aggregate
        district_stats = joined.groupby([admin_id_column, admin_name_column]).agg(
            mean_prob=(probability_column, 'mean'),
            max_prob=(probability_column, 'max'),
            min_prob=(probability_column, 'min'),
            std_prob=(probability_column, 'std'),
            point_count=(probability_column, 'count'),
            high_risk_count=(probability_column, lambda x: (x >= self.config.risk_thresholds.high_min).sum()),
            moderate_risk_count=(probability_column, lambda x: 
                ((x >= self.config.risk_thresholds.low_max) & (x < self.config.risk_thresholds.high_min)).sum()),
            low_risk_count=(probability_column, lambda x: (x < self.config.risk_thresholds.low_max).sum())
        ).reset_index()
        
        # Merge back with boundaries
        district_risk = admin_boundaries.merge(district_stats, on=admin_id_column, how='left')
        
        # Classify district risk
        district_risk['risk_level'] = pd.cut(
            district_risk['mean_prob'],
            bins=[0, self.config.risk_thresholds.low_max, 
                  self.config.risk_thresholds.moderate_max, 1.0],
            labels=['LOW', 'MODERATE', 'HIGH']
        )
        
        # Risk score (0-100)
        district_risk['risk_score'] = (district_risk['mean_prob'] * 100).round(1)
        
        if output_path:
            district_risk.to_file(output_path, driver='GeoJSON')
            logger.info(f"District-level risk saved to {output_path}")
        
        return district_risk
    
    def road_risk_analysis(self,
                          gdf: gpd.GeoDataFrame,
                          probability_column: str,
                          road_network: gpd.GeoDataFrame,
                          buffer_distances: Optional[List[float]] = None,
                          output_path: Optional[str] = None) -> gpd.GeoDataFrame:
        """
        Analyze risk along road networks
        """
        buffer_distances = buffer_distances or self.config.gis.road_buffer_distances
        
        # Ensure same CRS
        if gdf.crs != road_network.crs:
            gdf = gdf.to_crs(road_network.crs)
        
        results = []
        
        for dist_km in buffer_distances:
            # Buffer roads
            road_buffer = road_network.copy()
            road_buffer['geometry'] = road_buffer.geometry.buffer(dist_km / 111)  # approx deg
            
            # Dissolve buffers
            buffer_union = unary_union(road_buffer.geometry)
            buffer_gdf = gpd.GeoDataFrame(geometry=[buffer_union], crs=road_network.crs)
            
            # Find points within buffer
            within_buffer = gdf[gdf.within(buffer_union)]
            
            if len(within_buffer) > 0:
                stats = {
                    'buffer_km': dist_km,
                    'total_points': len(within_buffer),
                    'mean_prob': within_buffer[probability_column].mean(),
                    'max_prob': within_buffer[probability_column].max(),
                    'high_risk_pct': (within_buffer[probability_column] >= self.config.risk_thresholds.high_min).mean() * 100,
                    'length_km': road_network.length.sum() / 1000  # approx
                }
            else:
                stats = {
                    'buffer_km': dist_km,
                    'total_points': 0,
                    'mean_prob': 0,
                    'max_prob': 0,
                    'high_risk_pct': 0,
                    'length_km': road_network.length.sum() / 1000
                }
            
            results.append(stats)
        
        road_risk_df = pd.DataFrame(results)
        
        if output_path:
            road_risk_df.to_csv(output_path, index=False)
            logger.info(f"Road risk analysis saved to {output_path}")
        
        return road_risk_df
    
    def safe_unsafe_routes(self,
                          gdf: gpd.GeoDataFrame,
                          probability_column: str,
                          road_network: gpd.GeoDataFrame,
                          risk_threshold: float = 0.5,
                          output_path: Optional[str] = None) -> gpd.GeoDataFrame:
        """
        Classify road segments as safe/unsafe based on nearby risk
        """
        # Ensure same CRS
        if gdf.crs != road_network.crs:
            gdf = gdf.to_crs(road_network.crs)
        
        # Buffer roads slightly to capture nearby points
        road_network = road_network.copy()
        road_network['buffer'] = road_network.geometry.buffer(0.001)  # ~100m
        
        # For each road segment, find max risk in buffer
        road_risks = []
        for idx, road in road_network.iterrows():
            nearby = gdf[gdf.within(road['buffer'])]
            if len(nearby) > 0:
                max_risk = nearby[probability_column].max()
                mean_risk = nearby[probability_column].mean()
            else:
                max_risk = 0
                mean_risk = 0
            
            road_risks.append({
                'road_id': idx,
                'max_risk': max_risk,
                'mean_risk': mean_risk,
                'classification': 'UNSAFE' if max_risk >= risk_threshold else 'SAFE',
                'geometry': road.geometry
            })
        
        routes = gpd.GeoDataFrame(road_risks, geometry='geometry', crs=road_network.crs)
        
        if output_path:
            routes.to_file(output_path, driver='GeoJSON')
            logger.info(f"Safe/unsafe routes saved to {output_path}")
        
        return routes
    
    def create_risk_layers(self,
                          predictions: np.ndarray,
                          coordinates: np.ndarray,
                          base_gdf: Optional[gpd.GeoDataFrame] = None,
                          output_dir: Optional[str] = None) -> Dict[str, Any]:
        """
        Create all risk layers at once
        """
        output_dir = Path(output_dir or self.config.paths.outputs)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create base GeoDataFrame
        if base_gdf is None:
            geometry = [Point(xy) for xy in coordinates]
            base_gdf = gpd.GeoDataFrame(
                {f'prob_{i}': predictions for i in range(predictions.shape[1]) if len(predictions.shape) > 1},
                geometry=geometry,
                crs=self.crs
            )
            if len(predictions.shape) == 1:
                base_gdf['probability'] = predictions
            else:
                base_gdf['probability'] = predictions[:, 1]  # Assume binary classification
        
        prob_col = 'probability'
        
        layers = {}
        
        # 1. Susceptibility map
        layers['susceptibility'] = self.create_susceptibility_map(
            base_gdf, prob_col,
            output_dir / 'susceptibility_map.geojson'
        )
        
        # 2. Risk heatmap
        layers['heatmap'] = self.create_risk_heatmap(
            base_gdf, prob_col,
            output_dir / 'risk_heatmap.geojson'
        )
        
        # 3. High-risk zones
        layers['high_risk_zones'] = self.identify_high_risk_zones(
            base_gdf, prob_col,
            output_path=output_dir / 'high_risk_zones.geojson'
        )
        
        # 4. Risk raster
        bounds = base_gdf.total_bounds
        raster_data, transform = self.create_risk_raster(
            base_gdf[prob_col].values,
            np.array([(geom.x, geom.y) for geom in base_gdf.geometry]),
            bounds=bounds,
            output_path=output_dir / 'risk_raster.tif'
        )
        layers['raster'] = {'data': raster_data, 'transform': transform}
        
        logger.info("All risk layers created successfully")
        return layers
