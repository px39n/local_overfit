"""
Figure 5: Accuracy-Sparsity Map Visualization

This module provides functions for:
1. Calculating data sparsity/density maps
2. Creating land masks from shapefiles
3. Plotting accuracy maps with sparsity contours
"""

from pathlib import Path
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from tqdm.auto import tqdm
from sklearn.neighbors import BallTree, KDTree

# Data directory
DATA_DIR = Path(__file__).parent.parent / "data"


def make_mask(ds, gdf, var_name=None):
    """
    Create a boolean mask for land areas based on shapefile.
    
    Args:
        ds: xarray Dataset with longitude and latitude coordinates
        gdf: GeoDataFrame with land boundaries
        var_name: Optional variable name to use from ds (unused)
        
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


def haversine_distance(lon1, lat1, lon2, lat2):
    """
    Calculate great-circle distance between two points (in km).
    Correctly handles crossing the date line (-180°/180° boundary).
    
    Args:
        lon1, lat1: Coordinates of first point(s) in degrees
        lon2, lat2: Coordinates of second point(s) in degrees
        
    Returns:
        Distance in kilometers
    """
    R = 6371  # Earth radius in km
    
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    
    # Handle date line crossing: choose shortest longitude path
    dlon = (dlon + np.pi) % (2 * np.pi) - np.pi
    
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    c = 2 * np.arcsin(np.sqrt(a))
    distance = R * c
    
    return distance


def calculate_data_sparsity(df, reference_idx, output_idx=None, radius=None, 
                            metric='haversine', verbose=1):
    """
    Calculate weighted data density for all output points based on reference points.
    
    Density is computed as the weighted count of stations within a search radius,
    normalized by the search area. Weights decrease linearly with distance.
    
    Args:
        df: DataFrame containing longitude and latitude columns
        reference_idx: Index of reference points to build tree from
        output_idx: Index of points to calculate density for (default: all)
        radius: Search radius in km (default: auto-inferred as 1/8 of spatial range)
        metric: Distance metric ('haversine' or 'euclidean')
        verbose: Print progress (0=silent, 1=progress bar)
    
    Returns:
        DataFrame with new column 'data_sparsity' containing weighted density scores
    """
    df = df.copy()
    
    if output_idx is None:
        output_idx = df.index
        
    # Get unique reference points
    ref_points = df.loc[reference_idx, ["longitude", "latitude"]].drop_duplicates().values
    
    # Auto-infer radius if not provided
    if radius is None:
        spatial_range = haversine_distance(
            ref_points[:, 0].min(), ref_points[:, 1].min(),
            ref_points[:, 0].max(), ref_points[:, 1].max()
        )
        radius = spatial_range / 8
        print(f"Auto-inferred spatial radius: {radius:.2f} km")
    
    # Build spatial tree
    if metric == 'haversine':
        tree = BallTree(np.radians(ref_points), metric='haversine')
        radius_for_search = radius / 6371.0  # Convert to radians
    elif metric == 'euclidean':
        tree = KDTree(ref_points)
        radius_for_search = radius / 111.32  # Convert to degrees
    else:
        raise ValueError("metric must be 'haversine' or 'euclidean'")
    
    # Get unique output points
    output_points = df.loc[output_idx, ["longitude", "latitude"]].drop_duplicates().values
    
    # Search circle area (km²)
    circle_area = np.pi * radius ** 2
    
    # Calculate weighted density for each point
    point_sparsity = {}
        
    for i in tqdm(range(len(output_points)), 
                  desc=f"Calculating weighted density ({metric})",
                  disable=not verbose):
        point = output_points[i]
        
        if metric == 'haversine':
            point_for_search = np.radians(point)
        else:
            point_for_search = point
        
        neighbors = tree.query_radius(point_for_search.reshape(1, -1), r=radius_for_search)[0]
        
        if len(neighbors) == 0:
            # No neighbors: use nearest neighbor
            distances_knn, indices_knn = tree.query(point_for_search.reshape(1, -1), k=1)
            neighbors = indices_knn[0]
        
        # Calculate real distances
        distances = haversine_distance(
            point[0], point[1],
            ref_points[neighbors, 0], ref_points[neighbors, 1]
        )
        
        # Weight: linear decay with distance
        distances = np.minimum(distances, radius)
        weights = (radius - distances) / radius
        
        # Weighted density = weighted count / area
        weighted_count = np.sum(weights)
        weighted_density = weighted_count / circle_area
        
        point_sparsity[tuple(point)] = weighted_density
        
    # Map sparsity scores back to all output points
    all_output_points = df.loc[output_idx, ["longitude", "latitude"]].values
    output_tuples = pd.MultiIndex.from_arrays(
        [all_output_points[:, 0], all_output_points[:, 1]], names=['lon', 'lat']
    )
    point_tuples = pd.MultiIndex.from_arrays(
        [output_points[:, 0], output_points[:, 1]], names=['lon', 'lat']
    )
    
    sparsity_map = pd.Series(point_sparsity, index=point_tuples)
    scores = sparsity_map.reindex(output_tuples, fill_value=0).values
    
    df.loc[output_idx, 'data_sparsity'] = scores
    return df


def calculate_sparsity_map(df, ds, reference_idx=None, radius=None, 
                           smooth_sigma=10, metric='haversine'):
    """
    Calculate sparsity map on a regular grid and return as xarray Dataset.
    
    Args:
        df: DataFrame containing station locations (longitude, latitude)
        ds: xarray Dataset defining the output grid coordinates
        reference_idx: Index of reference stations (default: all)
        radius: Search radius in km (default: auto-inferred)
        smooth_sigma: Gaussian filter sigma for smoothing (0 = no smoothing)
        metric: Distance metric ('haversine' or 'euclidean')
    
    Returns:
        xarray Dataset with 'sparsity' variable on the same grid as input ds
    """
    from scipy.ndimage import gaussian_filter
    
    if reference_idx is None:
        reference_idx = df.index
    
    if 'longitude' not in ds.coords or 'latitude' not in ds.coords:
        raise ValueError("Input dataset must contain 'longitude' and 'latitude' coordinates")
    
    lon = ds.longitude.values
    lat = ds.latitude.values
    
    # Create grid points DataFrame
    lon_grid, lat_grid = np.meshgrid(lon, lat)
    grid_points = pd.DataFrame({
        'longitude': lon_grid.flatten(),
        'latitude': lat_grid.flatten()
    })
    
    # Combine with original data
    df_temp = df.copy()
    grid_start_idx = len(df_temp)
    grid_indices = range(grid_start_idx, grid_start_idx + len(grid_points))
    grid_points.index = grid_indices
    df_combined = pd.concat([df_temp, grid_points], ignore_index=False)
    
    # Calculate sparsity for grid points
    df_result = calculate_data_sparsity(
        df_combined, 
        reference_idx=reference_idx, 
        output_idx=grid_indices, 
        radius=radius,
        metric=metric
    )
    
    # Extract and reshape
    sparsity_values = df_result.loc[grid_indices, 'data_sparsity'].values
    sparsity_grid = sparsity_values.reshape(len(lat), len(lon))
    
    # Apply smoothing
    if smooth_sigma > 0:
        sparsity_grid = gaussian_filter(sparsity_grid, sigma=smooth_sigma)
    
    # Create output Dataset
    ds_sparsity = xr.Dataset(
        data_vars=dict(sparsity=(["latitude", "longitude"], sparsity_grid)),
        coords=dict(longitude=lon, latitude=lat),
        attrs=dict(
            description="Data sparsity map based on monitoring station density",
            radius_km=radius if radius is not None else "auto-inferred",
            smooth_sigma=smooth_sigma,
            metric=metric
        )
    )
    
    return ds_sparsity


def plot_accuracy_sparsity_map(
    df_analysis,
    ds_sparsity,
    states,
    train_data,
    mask=None,
    model_list=['lightgbm', 'resnet+biglightgbm'],
    accuracy_range=(0, 1),
    levels=(0.25, 0.5, 0.75),
    sparsity_quantiles=3,
    sparsity_levels=None,
    model_type='two_stage',
    sufficiency=500000,
    sparsity_bins=7,
    full_features='Full',
    cmap_name='Spectral_r',
    figsize_per_plot=4,
    height=3,
    splitby="grid",
    outlier_threshold=None,
    resolution=None,
    save_path=None,
    show_plot=True
):
    """
    Create multi-panel plot showing station density and model accuracy.
    
    Args:
        df_analysis: DataFrame with predictions and sparsity columns
        ds_sparsity: xarray Dataset containing sparsity map
        states: GeoDataFrame with region boundaries for plotting
        train_data: DataFrame with training station locations
        mask: Optional mask for focus region
        model_list: List of model names to plot
        accuracy_range: (min, max) for colorbar
        levels: Contour levels for accuracy
        sparsity_quantiles: Number of sparsity quantiles to show
        sparsity_levels: Explicit sparsity contour levels (overrides quantiles)
        model_type: Visualization mode:
            - 'two_stage': TwoStageModel prediction
            - 'observation': Raw station R² values
            - 'average_NxM': Grid-averaged R² (e.g., 'average_15x20')
            - 'svm', 'gam', etc.: Fitted surface model
        sufficiency: Sample size for filtering
        sparsity_bins: Number of sparsity bins
        full_features: Feature set ('Full', 'Spatial', 'Density', or list)
        cmap_name: Matplotlib colormap name
        figsize_per_plot: Width per subplot (inches)
        height: Figure height (inches)
        splitby: Aggregation method ('grid' or 'station')
        outlier_threshold: R² outlier threshold
        resolution: [lon_bins, lat_bins] for spatial aggregation
        save_path: Path to save figure
        show_plot: Whether to display the plot
    
    Returns:
        fig, axes, fitted_models
    """
    from .figure5_others import (
        fit_universal_models,
        calculate_station_r2,
        plot_observation_points,
        plot_grid_average,
        plot_anomaly_stations,
        plot_sparsity_quantile_contour,
        plot_accuracy_hexbin,
        plot_multipolygon_edges,
        plot_station_points
    )
    
    # Filter to required columns
    base_cols = ['observed', 'sufficiency', 'sparsity', 'longitude', 'latitude']
    model_cols = [f'predicted_{model}' for model in model_list]
    required_cols = [c for c in base_cols + model_cols if c in df_analysis.columns]
    df_filtered = df_analysis[required_cols].copy()
    
    # Fit models (skip for observation/average modes)
    if model_type in ['observation', 'anomaly'] or model_type.startswith('average'):
        fitted_models = None
        print(f"Using {model_type} mode")
    else:
        fitted_models = fit_universal_models(
            df_filtered,
            model_type=model_type,
            sparsity_bins=sparsity_bins,
            verbose=False,
            splitby=splitby,
            full_features=full_features,
            outlier_threshold=outlier_threshold,
            resolution=resolution
        )
    
    # Setup colormap
    min_accuracy, max_accuracy = accuracy_range
    cmap = getattr(plt.cm, cmap_name)
    cmap.set_under('lightgray')
    
    # Create figure
    total_plots = len(model_list) + 1
    fig = plt.figure(figsize=(figsize_per_plot * total_plots, height))
    gs = gridspec.GridSpec(
        1, total_plots + 1,
        width_ratios=[1] * total_plots + [0.05],
        figure=fig,
        wspace=0.03
    )
    
    axes = [fig.add_subplot(gs[0, i]) for i in range(total_plots)]
    cax = fig.add_subplot(gs[0, -1])
    
    # Setup axes
    for i, ax in enumerate(axes):
        plot_multipolygon_edges(ax, states, edge_color='grey', linewidth=2, alpha=0.3)
        
        if i == 0:
            plot_station_points(ax, train_data, alpha=0.4)
        
        ax.set_xlabel('Longitude', fontsize=10)
        ax.set_ylabel('Latitude', fontsize=10) if i == 0 else ax.set_yticklabels([])
        ax.grid(True, alpha=0.2, linewidth=0.5)
        ax.set_xlim([ds_sparsity.longitude.min().item(), ds_sparsity.longitude.max().item()])
        ax.set_ylim([ds_sparsity.latitude.min().item(), ds_sparsity.latitude.max().item()])
        
        if i == 0:
            plot_sparsity_quantile_contour(
                ax, ds_sparsity,
                count_number=sparsity_quantiles,
                levels=sparsity_levels,
                colors=['black'] * sparsity_quantiles,
                linewidth=[2.5, 2, 1.5][:sparsity_quantiles],
                line_styles=['-'] * sparsity_quantiles,
                alpha=0.65,
                show_labels=True,
                label_format='Q{index:d}'
            )
    
    # Colorbar
    norm = plt.Normalize(min_accuracy, max_accuracy)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    
    # Plot each panel
    for i, ax in enumerate(axes):
        if i == 0:
            ax.set_title('Station Density', fontsize=10)
        else:
            model_name = model_list[i - 1]
            
            if model_type == 'observation':
                station_data = calculate_station_r2(df_analysis, model_name, sufficiency)
                plot_observation_points(ax, station_data, cmap, min_accuracy, max_accuracy, marker_size=15)
                
            elif model_type.startswith('average'):
                try:
                    grid_str = model_type.split('_')[1]
                    n_lat, n_lon = map(int, grid_str.split('x'))
                    grid_shape = (n_lat, n_lon)
                except:
                    grid_shape = (15, 20)
                plot_grid_average(ax, df_analysis, model_name, sufficiency, cmap, 
                                 min_accuracy, max_accuracy, grid_shape=grid_shape)
                
            elif model_type == 'anomaly':
                station_data = calculate_station_r2(df_analysis, model_name, sufficiency)
                plot_anomaly_stations(ax, station_data, marker_size=30, marker_color='red')
                
            else:
                ds_to_plot = ds_sparsity.where(mask) if mask is not None else ds_sparsity
                plot_accuracy_hexbin(
                    ax, fitted_models, model_name,
                    ds_to_plot,
                    sufficiency,
                    cmap=cmap,
                    vmin=min_accuracy,
                    vmax=max_accuracy,
                    add_colorbar=False,
                    levels=list(levels),
                    contour_alpha=0.65,
                    contour_linewidths=[1.5, 2, 2.5][:len(levels)],
                    colorbar_label='Accuracy',
                    alpha=0.945
                )
            
            ax.set_title(model_name, fontsize=10)
    
    plt.colorbar(sm, cax=cax, label='Accuracy (R²)', extend='min')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, format='svg' if save_path.endswith('.svg') else 'png', dpi=300)
        print(f"Figure saved to: {save_path}")
    
    if show_plot:
        plt.show()
    
    return fig, axes, fitted_models
