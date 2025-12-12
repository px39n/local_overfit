"""
Utility functions for data loading and processing.
"""

import os
import pickle as pkl
import pandas as pd
import numpy as np
import xarray as xr
import geopandas as gpd
from pathlib import Path
from tqdm.auto import tqdm
from sklearn.neighbors import BallTree, KDTree

# =============================================================================
# Data Paths
# =============================================================================

DATA_DIR = Path(__file__).parent.parent / "data"

def get_data_path(relative_path: str) -> Path:
    """Get absolute path from relative data path."""
    return DATA_DIR / relative_path

# =============================================================================
# Geographic Distance Functions
# =============================================================================

def haversine_distance(lon1, lat1, lon2, lat2):
    """
    Calculate great-circle distance between two points (in km).
    Correctly handles crossing the date line (-180°/180° boundary).
    
    Args:
        lon1, lat1: Longitude and latitude of first point(s)
        lon2, lat2: Longitude and latitude of second point(s)
        
    Returns:
        Distance in kilometers
    """
    R = 6371  # Earth radius in km
    
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    
    # Handle crossing date line: choose shortest longitude path
    dlon = (dlon + np.pi) % (2*np.pi) - np.pi
    
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    distance = R * c
    
    return distance

# =============================================================================
# Data Density/Sparsity Calculation
# =============================================================================

def calculate_data_sparsity(df, reference_idx, output_idx=None, radius=None, 
                            metric='haversine', verbose=1):
    """
    Calculate weighted data sparsity score for all output points based on reference points.
    
    This function computes a density score for each point based on the number and 
    distance of nearby reference points within a search radius.
    
    Args:
        df: DataFrame containing 'longitude' and 'latitude' columns
        reference_idx: Index of reference points to build tree from
        output_idx: Optional index of points to calculate sparsity for. 
                   If None, calculates for all points in df
        radius: Radius in km to search for neighbors. 
               If None, automatically inferred as 1/8 of spatial range
        metric: Distance metric to use ('haversine' or 'euclidean')
        verbose: Whether to show progress bar (1=show, 0=hide)
    
    Returns:
        DataFrame with new column 'data_sparsity' containing weighted density scores
    
    Example:
        >>> df_with_density = calculate_data_sparsity(
        ...     df, 
        ...     reference_idx=train_indices, 
        ...     output_idx=test_indices,
        ...     radius=500,  # 500 km
        ...     verbose=1
        ... )
        >>> density = df_with_density.loc[test_indices, 'data_sparsity']
    """
    df = df.copy()
    
    if output_idx is None:
        output_idx = df.index
        
    # Get unique reference points for building tree
    ref_points = df.loc[reference_idx, ["longitude", "latitude"]].drop_duplicates().values
    
    # Auto-infer radius if not provided
    if radius is None:
        spatial_range = haversine_distance(
            ref_points[:, 0].min(), ref_points[:, 1].min(),
            ref_points[:, 0].max(), ref_points[:, 1].max()
        )
        radius = spatial_range / 8
        if verbose:
            print(f"Auto-inferred spatial radius: {radius:.2f} km")
    
    # Build tree based on metric
    if metric == 'haversine':
        tree = BallTree(np.radians(ref_points), metric='haversine')
        radius_for_search = radius / 6371.0  # Convert to radians
    elif metric == 'euclidean':
        tree = KDTree(ref_points)
        radius_for_search = radius / 111.32  # Convert to degrees
    else:
        raise ValueError("metric must be 'haversine' or 'euclidean'")
    
    # Get unique output points to calculate sparsity for
    output_points = df.loc[output_idx, ["longitude", "latitude"]].drop_duplicates().values
    
    # Calculate area of the search circle (in km²)
    circle_area = np.pi * radius**2
    
    # Calculate weighted density for each unique output point
    point_sparsity = {}
        
    for i in tqdm(range(len(output_points)), 
                  desc=f"Calculating weighted data density ({metric})",
                  disable=not verbose):
        point = output_points[i]
        
        if metric == 'haversine':
            point_for_search = np.radians(point)
        else:
            point_for_search = point
        
        neighbors = tree.query_radius(point_for_search.reshape(1, -1), r=radius_for_search)[0]
        
        if len(neighbors) == 0:
            # No neighbors, use nearest neighbor
            distances_knn, indices_knn = tree.query(point_for_search.reshape(1, -1), k=1)
            neighbors = indices_knn[0]
            
            distances = haversine_distance(
                point[0], point[1],
                ref_points[neighbors, 0], ref_points[neighbors, 1]
            )
        else:
            distances = haversine_distance(
                point[0], point[1],
                ref_points[neighbors, 0], ref_points[neighbors, 1]
            )
        
        # Calculate weights: weight = (radius - distance) / radius
        distances = np.minimum(distances, radius)
        weights = (radius - distances) / radius
        
        # Weighted count
        weighted_count = np.sum(weights)
        
        # Weighted density = weighted count / area
        weighted_density = weighted_count / circle_area
        
        point_sparsity[tuple(point)] = weighted_density
        
    # Assign sparsity scores to all points based on their coordinates
    all_output_points = df.loc[output_idx, ["longitude", "latitude"]].values
    output_tuples = pd.MultiIndex.from_arrays(
        [all_output_points[:, 0], all_output_points[:, 1]], 
        names=['lon', 'lat']
    )
    point_tuples = pd.MultiIndex.from_arrays(
        [output_points[:, 0], output_points[:, 1]], 
        names=['lon', 'lat']
    )
    
    # Create mapping dictionary
    sparsity_map = pd.Series(point_sparsity, index=point_tuples)
    
    # Use reindex for fast lookup
    scores = sparsity_map.reindex(output_tuples, fill_value=0).values
    
    df.loc[output_idx, 'data_sparsity'] = scores
    return df

# =============================================================================
# Data Loading Functions
# =============================================================================

def load_station_data() -> pd.DataFrame:
    """
    Load station observation data with features.
    
    Returns:
        pd.DataFrame: Station data with 22 features
    """
    path = get_data_path("training/Site_Observation_with_feature.pkl")
    df = pkl.load(open(path, 'rb'))
    df["time"] = pd.to_datetime(df["time"])
    for col in df.select_dtypes(include=["float64", "int64"]).columns:
        df[col] = df[col].astype("float32")
    return df

def load_ozone_grid() -> xr.Dataset:
    """
    Load gridded ozone data.
    
    Returns:
        xr.Dataset: Ozone data at 10km resolution
    """
    path = get_data_path("ozone.nc")
    return xr.open_dataset(path)

def load_map_boundaries() -> gpd.GeoDataFrame:
    """
    Load Natural Earth land boundaries.
    
    Returns:
        gpd.GeoDataFrame: Land boundaries for Europe
    """
    path = get_data_path("ne_10m_land.shp")
    return gpd.read_file(path)

def load_cv_results(project_path: str) -> dict:
    """
    Load cross-validation results from project directory.
    
    Args:
        project_path: Path to CV results directory
        
    Returns:
        dict: CV results for each fold and model
    """
    import json
    path = get_data_path(project_path)
    meta_path = path / "meta_split.json"
    
    with open(meta_path, 'r') as f:
        return json.load(f)

def load_spatial_prediction(model_name: str) -> xr.Dataset:
    """
    Load spatial prediction NetCDF file.
    
    Args:
        model_name: Name of the model (e.g., 'lightgbm', 'mlp_og')
        
    Returns:
        xr.Dataset: Spatial prediction data
    """
    path = get_data_path(f"appendix/geo_prediction/{model_name}.nc")
    return xr.open_dataset(path)

def load_analysis_data(sufficiency: int = None) -> pd.DataFrame:
    """
    Load accuracy analysis data.
    
    Args:
        sufficiency: Filter by sample size (e.g., 10000, 1000000)
        
    Returns:
        pd.DataFrame: Analysis data
    """
    path = get_data_path("Analysis/validation/df_analysis.pkl")
    df = pkl.load(open(path, 'rb'))
    
    if sufficiency is not None:
        df = df[df["sufficiency"] == sufficiency]
    
    return df

# =============================================================================
# Data Processing Functions
# =============================================================================

def calculate_sparsity_map(df, ds, reference_idx=None, radius=None, smooth_sigma=10, 
                          metric='haversine'):
    """
    Calculate sparsity map and return as xarray Dataset.
    
    Args:
        df: DataFrame containing longitude and latitude columns
        ds: xarray Dataset with longitude and latitude coordinates (used as reference for output grid)
        reference_idx: Optional index of reference points. If None, uses all points in df
        radius: Radius in km to search for neighbors. If None, automatically inferred
        smooth_sigma: Gaussian filter sigma for smoothing. If 0, no smoothing applied
        metric: Distance metric to use ('haversine' or 'euclidean')
    
    Returns:
        xarray Dataset with sparsity values on the same grid as input ds
    """
    from scipy.ndimage import gaussian_filter
    
    if reference_idx is None:
        reference_idx = df.index
    
    # Check if ds has required coordinates
    if 'longitude' not in ds.coords or 'latitude' not in ds.coords:
        raise ValueError("Input dataset must contain 'longitude' and 'latitude' coordinates")
    
    # Get coordinates from ds
    lon = ds.longitude.values
    lat = ds.latitude.values
    
    # Create a temporary DataFrame with all grid points
    lon_grid, lat_grid = np.meshgrid(lon, lat)
    grid_points = pd.DataFrame({
        'longitude': lon_grid.flatten(),
        'latitude': lat_grid.flatten()
    })
    
    # Add the grid points to the original dataframe temporarily
    df_temp = df.copy()
    grid_start_idx = len(df_temp)
    
    # Assign new indices to grid points
    grid_indices = range(grid_start_idx, grid_start_idx + len(grid_points))
    grid_points.index = grid_indices
    
    # Combine dataframes
    df_combined = pd.concat([df_temp, grid_points], ignore_index=False)
    
    # Calculate sparsity for grid points using existing function
    df_result = calculate_data_sparsity(
        df_combined, 
        reference_idx=reference_idx, 
        output_idx=grid_indices, 
        radius=radius,
        metric=metric,
        verbose=1
    )
    
    # Extract sparsity values for grid points
    sparsity_values = df_result.loc[grid_indices, 'data_sparsity'].values
    
    # Reshape to match grid
    sparsity_grid = sparsity_values.reshape(len(lat), len(lon))
    
    # Apply Gaussian smoothing if requested
    if smooth_sigma > 0:
        sparsity_grid = gaussian_filter(sparsity_grid, sigma=smooth_sigma)
    
    # Create output Dataset
    ds_sparsity = xr.Dataset(
        data_vars=dict(
            sparsity=(["latitude", "longitude"], sparsity_grid)
        ),
        coords=dict(
            longitude=lon,
            latitude=lat
        ),
        attrs=dict(
            description="Data sparsity map based on monitoring station density",
            radius_km=radius if radius is not None else "auto-inferred",
            smooth_sigma=smooth_sigma,
            metric=metric
        )
    )
    
    return ds_sparsity


def make_mask(ds, gdf, var_name=None):
    """
    Create a boolean mask for land areas based on shapefile.
    
    Args:
        ds: xarray Dataset with longitude and latitude coordinates
        gdf: GeoDataFrame with land boundaries
        var_name: Optional variable name to use from ds
        
    Returns:
        xarray DataArray with True for land areas
    """
    from shapely.geometry import Point
    
    lon = ds.longitude.values
    lat = ds.latitude.values
    
    # Create meshgrid
    lon_grid, lat_grid = np.meshgrid(lon, lat)
    
    # Flatten for point-in-polygon check
    points = [Point(x, y) for x, y in zip(lon_grid.flatten(), lat_grid.flatten())]
    
    # Check which points are within any polygon
    combined_geometry = gdf.unary_union
    mask_flat = np.array([combined_geometry.contains(p) for p in points])
    
    # Reshape to grid
    mask_grid = mask_flat.reshape(len(lat), len(lon))
    
    # Create DataArray
    mask_da = xr.DataArray(
        mask_grid,
        dims=["latitude", "longitude"],
        coords={"latitude": lat, "longitude": lon}
    )
    
    return mask_da


def split_by_region(df: pd.DataFrame, regions: list) -> tuple:
    """
    Split data by geographic regions.
    
    Args:
        df: DataFrame with latitude/longitude
        regions: List of [lat_min, lon_min, lat_max, lon_max]
        
    Returns:
        tuple: (train_indices, test_indices)
    """
    test_mask = pd.Series(False, index=df.index)
    
    for region in regions:
        lat_min, lon_min, lat_max, lon_max = region
        region_mask = (
            (df['latitude'] >= lat_min) & (df['latitude'] <= lat_max) &
            (df['longitude'] >= lon_min) & (df['longitude'] <= lon_max)
        )
        test_mask = test_mask | region_mask
    
    test_indices = df.index[test_mask]
    train_indices = df.index[~test_mask]
    
    return train_indices, test_indices


def prepare_figure4c_data(data_dir=None, time_range=None, sample=100000, seed=76, 
                          test_region=None):
    """
    Prepare data for Figure 4c with train/test split.
    
    Matches original parameters from 4. All_Figures.ipynb.
    
    Args:
        data_dir: Path to data directory
        time_range: [start_date, end_date] for filtering
        sample: Number of samples to use
        seed: Random seed
        test_region: List of test regions [[lat_min, lon_min, lat_max, lon_max], ...]
    
    Returns:
        tuple: (df, train_ind, test_ind)
    """
    if data_dir is None:
        data_dir = DATA_DIR
    else:
        data_dir = Path(data_dir)
    
    # Default parameters matching original code
    if time_range is None:
        time_range = ['2019-09-01', '2019-10-31']
    if test_region is None:
        test_region = [
            [60, 10, 75, 35], 
            [50, -10, 60, 1], 
            [45, 7, 55, 11], 
            [35, 36, 45, 41]
        ]
    
    # Load data
    df = pkl.load(open(data_dir / 'training/Site_Observation_with_feature.pkl', 'rb'))
    df["time"] = pd.to_datetime(df["time"])
    for col in df.select_dtypes(include=["float64", "int64"]).columns:
        df[col] = df[col].astype("float32")
    
    # Filter by time range
    time_mask = (df["time"] >= time_range[0]) & (df["time"] <= time_range[1])
    df = df[time_mask].copy()
    
    # Sample data
    np.random.seed(seed)
    if len(df) > sample:
        sample_idx = np.random.choice(df.index, size=sample, replace=False)
        df = df.loc[sample_idx]
    
    # Split by region
    train_ind, test_ind = split_by_region(df, test_region)
    
    return df, train_ind, test_ind

