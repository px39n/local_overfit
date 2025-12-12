"""
Figure 1: Data Overview

Subplots:
- (a) Spatial distribution of TOAR stations
- (b) Density distribution (observed vs grid)
- (c) Validation results comparison
- (c inset) Density vs R² relationship

Output files:
- Figure/F1_data.png (combined figure for paper)
"""

import os
import json
import pickle as pkl
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import xarray as xr
import geopandas as gpd
from typing import Tuple, Optional, List
from pathlib import Path
from sklearn.metrics import r2_score
from matplotlib.ticker import MaxNLocator, FuncFormatter
import time

# Cartopy imports (optional, for geographic projections)
try:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    HAS_CARTOPY = True
except ImportError:
    HAS_CARTOPY = False
    print("Warning: cartopy not installed. plot_station_distribution will use simple projection.")

# =============================================================================
# Constants
# =============================================================================

DATA_DIR = Path(__file__).parent.parent / "data"
FIGURE_DIR = Path(__file__).parent.parent / "Figure"

# Ensure Figure directory exists
FIGURE_DIR.mkdir(parents=True, exist_ok=True)

# Strategy name mapping
STRATEGY_MAPPING = {
    'train': 'Train',
    'Sample': 'Time',  
    'Site': 'Site',
    'Grid': 'Region',
    'Spatiotemporal_block': 'Region&Time',
}

# Model colors
MODEL_COLORS = {
    'mlp': '#2E86AB',       # Blue
    'mlp_early': '#2E86AB', # Blue
    'lightgbm': '#A23B72',  # Purple/Pink
}

# =============================================================================
# Figure 1a: Station Spatial Distribution
# =============================================================================

def plot_station_distribution(
    df: pd.DataFrame = None,
    ds_sparsity: xr.Dataset = None,
    states: gpd.GeoDataFrame = None,
    mask: xr.DataArray = None,
    figsize: Tuple[float, float] = (8, 6),
    cmap: str = "plasma",
    projection: str = 'LambertConformal',
    central_longitude: float = 10.0,
    central_latitude: float = 50.0,
    station_color: str = 'white',
    station_size: float = 0.8,
    boundary_simplify: float = 0.05,
    raster_factor: int = 2,
    extent: list = None,
    layout: str = 'horizontal',
    contour: float = None,
    contour_color: str = 'white',
    contour_linewidth: float = 2.5,
    title: str = 'European Ozone Monitoring Network',
    colorbar_label: str = 'Station Density (stations/100km²)',
    horizontal_colorbar_tick_size: int = 8,
    horizontal_colorbar_width: float = 0.76,
    horizontal_other_font_size: int = 12,
    show_progress: bool = True,
    save_path: Optional[str] = None,
    dpi: int = 300
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plot spatial distribution of monitoring stations with density background.
    
    Args:
        df: Station data with latitude/longitude. If None, load from default path.
        ds_sparsity: Sparsity dataset. If None, calculate from df.
        states: Land boundaries GeoDataFrame. If None, load from default path.
        mask: Land mask DataArray. If None, calculate from states.
        figsize: Figure size
        cmap: Colormap for density
        projection: Map projection ('LambertConformal', 'PlateCarree', etc.)
        central_longitude: Center longitude for projection
        central_latitude: Center latitude for projection
        station_color: Color for station markers
        station_size: Size of station markers
        boundary_simplify: Tolerance for boundary simplification (0 = no simplification)
        raster_factor: Downsampling factor for raster
        extent: Map extent [lon_min, lon_max, lat_min, lat_max]
        layout: 'vertical' or 'horizontal' colorbar layout
        contour: Contour line value (None = no contour)
        contour_color: Color for contour line
        contour_linewidth: Width of contour line
        title: Plot title
        colorbar_label: Label for colorbar
        horizontal_colorbar_tick_size: Colorbar tick font size (horizontal layout)
        horizontal_colorbar_width: Colorbar width ratio (horizontal layout)
        horizontal_other_font_size: Other font size (horizontal layout)
        show_progress: Whether to show timing progress
        save_path: Path to save figure
        dpi: Resolution for saving
        
    Returns:
        Tuple of (Figure, Axes, ColorbarAxes)
    """
    from .utils import calculate_sparsity_map, make_mask
    
    # Load data if not provided
    if df is None:
        df = pkl.load(open(DATA_DIR / "training/Site_Observation_with_feature.pkl", 'rb'))
        df["time"] = pd.to_datetime(df["time"])
    
    if states is None:
        states = gpd.read_file(DATA_DIR / "ne_10m_land.shp")
    
    # Calculate sparsity map if not provided
    if ds_sparsity is None:
        ds_ozone = xr.open_dataset(DATA_DIR / "ozone.nc")
        print("Calculating sparsity map...")
        ds_sparsity = calculate_sparsity_map(
            df, ds_ozone, reference_idx=None, radius=500, 
            smooth_sigma=10, metric="haversine"
        )
        # Extend coordinates
        ds_sparsity = ds_sparsity.reindex(
            longitude=np.arange(ds_sparsity.longitude.min(), 60.1, 0.25),
            latitude=np.arange(30, ds_sparsity.latitude.max(), 0.25),
            method='nearest'
        )
    
    # Calculate mask if not provided
    if mask is None:
        mask = make_mask(ds_sparsity, states)
    
    if extent is None:
        extent = [-10, 40, 35, 65]
    
    # Check if cartopy is available
    if not HAS_CARTOPY:
        # Fallback to simple plot without projection
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot land boundaries
        states.boundary.plot(ax=ax, linewidth=0.5, color='black')
        
        # Get unique stations
        unique_stations = df[['longitude', 'latitude']].drop_duplicates()
        
        # Plot density background
        sparsity_masked = ds_sparsity.sparsity.where(mask) * 10000
        sparsity_masked.plot(ax=ax, cmap=cmap, add_colorbar=True, alpha=0.8)
        
        # Plot stations
        ax.scatter(unique_stations['longitude'], unique_stations['latitude'],
                   c=station_color, s=station_size, alpha=0.7)
        
        ax.set_xlim(extent[0], extent[1])
        ax.set_ylim(extent[2], extent[3])
        ax.set_title(title)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
            print(f"✓ Figure saved to: {save_path}")
        
        return fig, ax, None
    
    # Full implementation with cartopy
    font_size = 12
    title_size = 14
    label_size = 10
    
    # Select projection
    if projection == 'LambertConformal':
        proj = ccrs.LambertConformal(central_longitude=central_longitude, 
                                     central_latitude=central_latitude)
    elif projection == 'Mercator':
        proj = ccrs.Mercator(central_longitude=central_longitude)
    else:
        proj = ccrs.PlateCarree(central_longitude=central_longitude)
    
    # Create figure with GridSpec
    fig = plt.figure(figsize=figsize)
    
    if layout == 'horizontal':
        cbar_width = horizontal_colorbar_width if horizontal_colorbar_width else 1
        gs = gridspec.GridSpec(1, 2, width_ratios=[cbar_width, 20], wspace=0.25)
        cax = fig.add_subplot(gs[0])
        ax = fig.add_subplot(gs[1], projection=proj)
    else:
        gs = gridspec.GridSpec(2, 1, height_ratios=[20, 1], hspace=0.25)
        ax = fig.add_subplot(gs[0], projection=proj)
        cax = fig.add_subplot(gs[1])
    
    start_time = time.time()
    
    if show_progress:
        print("🕐 Starting plot generation...")
    
    # Apply mask to sparsity data
    sparsity_masked = ds_sparsity.sparsity.where(mask) * 10000
    
    # Raster downsampling
    if raster_factor > 1:
        sparsity_masked = sparsity_masked.coarsen(
            longitude=raster_factor, latitude=raster_factor, boundary='trim'
        ).mean()
    
    # Plot density
    im = sparsity_masked.plot(
        ax=ax, cmap=cmap, transform=ccrs.PlateCarree(),
        add_colorbar=False, alpha=0.8, rasterized=True
    )
    
    # Add contour if specified
    if contour is not None:
        ax.contour(
            sparsity_masked.longitude, sparsity_masked.latitude,
            sparsity_masked.values, levels=[contour],
            colors=[contour_color], linestyles=['--'],
            linewidths=contour_linewidth, transform=ccrs.PlateCarree(), zorder=4
        )
    
    # Add colorbar
    cbar_orientation = 'vertical' if layout == 'horizontal' else 'horizontal'
    cbar = plt.colorbar(im, cax=cax, orientation=cbar_orientation, pad=0.05)
    
    if layout == 'horizontal':
        cbar_tick_size = horizontal_colorbar_tick_size or label_size
        cbar.set_label(colorbar_label, fontsize=horizontal_other_font_size or font_size)
        cbar.ax.tick_params(labelsize=cbar_tick_size)
        cbar.ax.yaxis.set_ticks_position('left')
        cbar.ax.yaxis.set_label_position('left')
    else:
        cbar.set_label(colorbar_label, fontsize=font_size)
        cbar.ax.tick_params(labelsize=label_size)
    
    # Add station points
    unique_stations = df[['longitude', 'latitude']].drop_duplicates()
    station_scatter = ax.scatter(
        unique_stations['longitude'], unique_stations['latitude'],
        c=station_color, s=station_size, alpha=0.7,
        transform=ccrs.PlateCarree(),
        label=f'Observations', zorder=5
    )
    
    # Plot boundaries
    if boundary_simplify is not None:
        simplified_states = states.simplify(boundary_simplify) if boundary_simplify > 0 else states
        simplified_states.boundary.plot(
            ax=ax, color='black', linewidth=0.8,
            transform=ccrs.PlateCarree(), zorder=3, rasterized=True
        )
    
    # Add map features
    ax.add_feature(cfeature.COASTLINE.with_scale('50m'), linewidth=0.5, alpha=0.7)
    ax.add_feature(cfeature.BORDERS.with_scale('50m'), linewidth=0.3, alpha=0.7)
    ax.add_feature(cfeature.LAND.with_scale('110m'), alpha=0.1, color='lightgray')
    ax.add_feature(cfeature.OCEAN.with_scale('110m'), alpha=0.1, color='lightblue')
    
    # Set extent
    ax.set_extent(extent, crs=ccrs.PlateCarree())
    
    # Add gridlines
    gl = ax.gridlines(draw_labels=True, alpha=0.4, linestyle='--', linewidth=0.8,
                      color='gray', x_inline=False, y_inline=False)
    gl.top_labels = False
    gl.right_labels = False
    gl.xlabel_style = {'size': horizontal_other_font_size or label_size}
    gl.ylabel_style = {'size': horizontal_other_font_size or label_size}
    
    # Title and legend
    ax.set_title(title, fontsize=horizontal_other_font_size or title_size, fontweight='bold', pad=20)
    
    legend = ax.legend(loc='upper right', bbox_to_anchor=(0.98, 0.98),
                       frameon=True, fancybox=True, 
                       fontsize=horizontal_other_font_size or font_size,
                       facecolor='white', edgecolor='black', framealpha=0.9,
                       markerscale=3.0)
    
    # Adjust colorbar alignment
    plt.draw()
    ax_pos_final = ax.get_position()
    cax_pos_final = cax.get_position()
    if layout == 'horizontal':
        cax.set_position([cax_pos_final.x0, ax_pos_final.y0, cax_pos_final.width, ax_pos_final.height])
    else:
        cax.set_position([ax_pos_final.x0, cax_pos_final.y0, ax_pos_final.width, cax_pos_final.height])
    
    if show_progress:
        print(f"🏁 Total time: {time.time() - start_time:.2f}s")
    
    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
        print(f"✓ Figure saved to: {save_path}")
    
    return fig, ax, cax

# =============================================================================
# Figure 1b: Density Distribution Comparison
# =============================================================================

def plot_density_comparison(
    df: pd.DataFrame = None,
    ds_sparsity: xr.Dataset = None,
    mask: xr.DataArray = None,
    density_col: str = 'data_sparsity',
    bins: int = 50,
    kde: bool = True,
    hist: bool = True,
    scale_factor: float = 10000,
    x_label: str = None,
    y_label: str = 'Relative Frequency',
    title: str = 'Observed vs Under Prediction',
    labels: Tuple[str, str] = ('Observation', 'Prediction'),
    colors: Tuple[str, str] = None,
    alpha: float = 0.6,
    figsize: Tuple[float, float] = (7, 2),
    font_size: int = 16,
    x_lim: Tuple[float, float] = None,
    y_lim: Tuple[float, float] = None,
    x_lim_number: int = None,
    y_lim_number: int = None,
    legend: bool = True,
    show_title: bool = False,
    show_x_label: bool = False,
    show_y_label: bool = False,
    radius: float = 500,
    save_path: Optional[str] = None
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plot density distribution comparison between observed stations and prediction regions.
    
    Args:
        df: DataFrame containing observed station data. If None, load and calculate.
        ds_sparsity: Dataset with gridded sparsity values. If None, calculate.
        mask: Boolean mask for land areas. If None, calculate.
        density_col: Column name in df containing density values
        bins: Number of histogram bins
        kde: Whether to show kernel density estimation curve
        hist: Whether to show histogram bars
        scale_factor: Scale factor for density values (e.g., 10000 for ×10⁴)
        x_label: X-axis label. If None, auto-generate.
        y_label: Y-axis label
        title: Plot title
        labels: Labels for (observed, prediction) distributions
        colors: Colors for (observed, prediction). If None, use defaults.
        alpha: Transparency for histogram bars
        figsize: Figure size
        font_size: Base font size
        x_lim: X-axis limits
        y_lim: Y-axis limits
        x_lim_number: Number of ticks on X-axis
        y_lim_number: Number of ticks on Y-axis
        legend: Whether to show legend
        show_title: Whether to show title
        show_x_label: Whether to show X-axis label
        show_y_label: Whether to show Y-axis label
        radius: Radius for density calculation (km)
        save_path: Path to save figure
        
    Returns:
        Tuple of (Figure, Axes)
    """
    from .utils import calculate_data_sparsity, calculate_sparsity_map, make_mask
    
    # Load and prepare data if not provided
    if df is None:
        df = pkl.load(open(DATA_DIR / "training/Site_Observation_with_feature.pkl", 'rb'))
        df["time"] = pd.to_datetime(df["time"])
    
    # Calculate station density if not in dataframe
    if density_col not in df.columns:
        print(f"Calculating station density (radius={radius}km)...")
        df = calculate_data_sparsity(
            df, reference_idx=df.index, output_idx=df.index, 
            radius=radius, verbose=1
        )
    
    # Calculate sparsity map if not provided
    if ds_sparsity is None:
        ds_ozone = xr.open_dataset(DATA_DIR / "ozone.nc")
        print("Calculating sparsity map...")
        ds_sparsity = calculate_sparsity_map(
            df, ds_ozone, reference_idx=None, radius=radius,
            smooth_sigma=10, metric="haversine"
        )
        ds_sparsity = ds_sparsity.reindex(
            longitude=np.arange(ds_sparsity.longitude.min(), 60.1, 0.25),
            latitude=np.arange(30, ds_sparsity.latitude.max(), 0.25),
            method='nearest'
        )
    
    # Calculate mask if not provided
    if mask is None:
        states = gpd.read_file(DATA_DIR / "ne_10m_land.shp")
        mask = make_mask(ds_sparsity, states)
    
    # Extract observed station density
    observed_values = df[density_col].dropna().values
    observed_values_scaled = observed_values * scale_factor
    
    # Extract prediction region density
    sparsity_data = ds_sparsity.sparsity if hasattr(ds_sparsity, 'sparsity') else ds_sparsity
    mask_flat = mask.values.flatten()
    sparsity_flat = sparsity_data.values.flatten()
    prediction_values = sparsity_flat[mask_flat & ~np.isnan(sparsity_flat)]
    prediction_values_scaled = prediction_values * scale_factor
    
    # Default colors
    if colors is None:
        colors = ('#2E86AB', '#A23B72')  # Blue, Purple
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot observed station distribution
    if hist:
        ax.hist(observed_values_scaled, bins=bins, alpha=alpha, color=colors[0],
                label=labels[0], edgecolor='black', linewidth=0.5, density=True)
    
    if kde:
        try:
            from scipy.stats import gaussian_kde
            kde_observed = gaussian_kde(observed_values_scaled)
            x_range = np.linspace(observed_values_scaled.min(), 
                                  observed_values_scaled.max(), 200)
            kde_values = kde_observed(x_range)
            ax.plot(x_range, kde_values, color=colors[0], linewidth=2.5, alpha=0.8)
        except ImportError:
            print("Warning: scipy not available, skipping KDE")
    
    # Plot prediction region distribution
    if len(prediction_values_scaled) > 0:
        if hist:
            ax.hist(prediction_values_scaled, bins=bins, alpha=alpha, color=colors[1],
                    label=labels[1], edgecolor='black', linewidth=0.5, density=True)
        
        if kde:
            try:
                from scipy.stats import gaussian_kde
                kde_prediction = gaussian_kde(prediction_values_scaled)
                x_range_pred = np.linspace(prediction_values_scaled.min(),
                                           prediction_values_scaled.max(), 200)
                kde_values_pred = kde_prediction(x_range_pred)
                ax.plot(x_range_pred, kde_values_pred, color=colors[1], 
                        linewidth=2.5, alpha=0.8)
            except ImportError:
                pass
    
    # Formatting
    if x_label is None:
        if scale_factor == 10000:
            x_label = 'Data Density (×10⁴)'
        elif scale_factor == 1000:
            x_label = 'Data Density (×10³)'
        else:
            x_label = f'Data Density (×{scale_factor:.0f})'
    
    if show_x_label:
        ax.set_xlabel(x_label, fontsize=font_size)
    if show_y_label:
        ax.set_ylabel(y_label, fontsize=font_size)
    if show_title:
        ax.set_title(title, fontsize=font_size+2, fontweight='bold')
    
    if legend:
        ax.legend(loc='best', fontsize=font_size-1, framealpha=0.9)
    
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.tick_params(axis='both', labelsize=font_size-2)
    
    # Set custom number of ticks if specified
    if x_lim_number is not None:
        ax.xaxis.set_major_locator(MaxNLocator(nbins=x_lim_number-1, prune=None))
    
    if y_lim_number is not None:
        ax.yaxis.set_major_locator(MaxNLocator(nbins=y_lim_number-1, prune=None))
    
    # Format tick labels
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, p: f'{x:.1f}'))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda y, p: f'{y:.1f}'))
    
    # Set axis limits if provided
    if x_lim is not None:
        ax.set_xlim(x_lim[0], x_lim[1])
    if y_lim is not None:
        ax.set_ylim(y_lim[0], y_lim[1])
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Figure saved to: {save_path}")
    
    return fig, ax

# =============================================================================
# Figure 1c (main): Validation Results Comparison
# =============================================================================

def plot_validation_comparison(
    project_path: str = None,
    model1: str = "mlp_early",
    model2: str = "lightgbm",
    strategies: List[str] = None,
    uncertainty: bool = True,
    y_lim: Tuple[float, float] = (0.40, 1.0),
    name: Tuple[str, str] = ("Deep Learning", "Classical ML"),
    figsize: Tuple[float, float] = (8, 4.8),
    save_path: Optional[str] = None,
    ax: plt.Axes = None
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plot R² comparison across validation strategies for two models.
    
    Args:
        project_path: Path to CV results. If None, use default.
        model1: First model name (e.g., "mlp_early")
        model2: Second model name (e.g., "lightgbm")
        strategies: List of validation strategies
        uncertainty: Whether to show uncertainty bands
        y_lim: Y-axis limits
        name: Display names for models
        figsize: Figure size
        save_path: Path to save figure
        ax: Optional axes to plot on
        
    Returns:
        Tuple of (Figure, Axes)
    """
    if project_path is None:
        project_path = str(DATA_DIR / "appendix/represent_10CV_100k")
    
    if strategies is None:
        strategies = ["train", "Sample", "Site", "Spatiotemporal_block"]
    
    # Map strategies to display names
    display_order = [STRATEGY_MAPPING.get(s, s) for s in strategies]
    
    # Model styling
    display_name1, display_name2 = name
    model_styles = {
        model1: {'color': '#2E86AB', 'marker': 'o', 'linestyle': '-', 'label': display_name1},
        model2: {'color': '#A23B72', 'marker': 's', 'linestyle': '--', 'label': display_name2}
    }
    
    # Create figure or use provided axes
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure
    
    # Process both models
    all_data = {}
    
    for model_name in [model1, model2]:
        csv_file = os.path.join(project_path, f"{model_name}.csv")
        
        if not os.path.exists(csv_file):
            print(f"⚠️  Warning: {csv_file} not found, skipping {model_name}")
            continue
        
        df = pd.read_csv(csv_file)
        
        # Filter valid data
        if "r2_score" in df.columns:
            df_clean = df[df['r2_score'].notna()].copy()
        else:
            df_clean = df.copy()
        
        if len(df_clean) == 0:
            continue
        
        # Map strategy names
        df_clean['display_name'] = df_clean['validation'].map(STRATEGY_MAPPING)
        df_clean = df_clean.dropna(subset=['display_name'])
        
        # Calculate statistics
        if 'fold' in df_clean.columns:
            grouped = df_clean.groupby('display_name')['r2_score'].agg(['mean', 'std', 'count'])
            grouped['std'] = grouped['std'].fillna(0)
        else:
            grouped = df_clean.groupby('display_name')['r2_score'].agg(['mean'])
            grouped['std'] = 0
            grouped['count'] = 1
        
        # Store ordered data
        ordered_means = []
        ordered_stds = []
        ordered_labels = []
        
        for display_name in display_order:
            if display_name in grouped.index:
                ordered_means.append(grouped.loc[display_name, 'mean'])
                ordered_stds.append(grouped.loc[display_name, 'std'] if 'std' in grouped.columns else 0)
                ordered_labels.append(display_name)
        
        all_data[model_name] = {
            'means': ordered_means,
            'stds': ordered_stds,
            'labels': ordered_labels,
            'has_folds': 'fold' in df_clean.columns
        }
    
    # Use labels from first available model
    if len(all_data) == 0:
        ax.text(0.5, 0.5, 'No data available', ha='center', va='center', transform=ax.transAxes)
        return fig, ax
    
    x_labels = all_data[list(all_data.keys())[0]]['labels']
    x_positions = np.arange(len(x_labels))
    
    # Plot each model
    for model_name, data in all_data.items():
        style = model_styles[model_name]
        means = data['means']
        stds = data['stds']
        
        # Main line and markers
        ax.plot(x_positions, means, 
               marker=style['marker'],
               linestyle=style['linestyle'],
               color=style['color'],
               linewidth=2.5,
               markersize=10,
               label=style['label'],
               alpha=0.9,
               zorder=3)
        
        # Uncertainty bands
        if uncertainty and data['has_folds']:
            ax.fill_between(x_positions, 
                           np.array(means) - np.array(stds),
                           np.array(means) + np.array(stds),
                           alpha=0.2,
                           color=style['color'],
                           zorder=1)
            ax.errorbar(x_positions, means, yerr=stds,
                       fmt='none',
                       ecolor=style['color'],
                       elinewidth=2,
                       capsize=5,
                       alpha=0.5,
                       zorder=2)
    
    # Formatting
    ax.set_xticks(x_positions)
    ax.set_xticklabels(x_labels, rotation=45, ha='right', fontsize=11)
    ax.set_ylabel('R² Score', fontsize=13, fontweight='bold')
    
    if y_lim is not None:
        ax.set_ylim(y_lim)
    
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.8)
    ax.legend(loc='best', fontsize=11, framealpha=0.9)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Figure saved to: {save_path}")
    
    return fig, ax

# =============================================================================
# Figure 1c (inset): Density vs R² Relationship
# =============================================================================

def plot_density_vs_r2(
    project_path: str = None,
    df: pd.DataFrame = None,
    model1: str = "mlp_early",
    model2: str = "lightgbm",
    radius: float = 500,
    name: Tuple[str, str] = ("Neural Network", "Tree-based Model"),
    sample_size: int = 100000,
    strategies: List[str] = None,
    uncertainty: bool = True,
    global_reference: bool = True,
    n_bins: int = 10,
    visual_point_bin: int = 50,
    show_title: bool = False,
    legend: bool = False,
    x_label: bool = False,
    y_label: bool = False,
    figsize: Tuple[float, float] = (12, 2.53),
    font_size: int = 15,
    x_lim_number: int = 4,
    y_lim_number: int = 5,
    save_path: Optional[str] = None
) -> Tuple[plt.Figure, List[plt.Axes]]:
    """
    Plot relationship between data density and model R².
    
    Args:
        project_path: Path to CV results. If None, use default.
        df: Full dataframe with station data. If None, load from default.
        model1: First model name
        model2: Second model name
        radius: Search radius in km
        name: Display names for models
        sample_size: Sample size filter
        strategies: Validation strategies to plot
        uncertainty: Whether to show uncertainty bands
        global_reference: Use global sparsity calculation
        n_bins: Number of bins for trend line
        visual_point_bin: Number of bins for scatter points
        show_title: Show subplot titles
        legend: Show legend
        x_label: Show x-axis label
        y_label: Show y-axis label
        figsize: Figure size
        font_size: Font size
        x_lim_number: Max x-axis tick intervals
        y_lim_number: Max y-axis tick intervals
        save_path: Path to save figure
        
    Returns:
        Tuple of (Figure, Axes list)
    """
    # Import sparsity calculation from local utils
    from .utils import calculate_data_sparsity
    
    if project_path is None:
        project_path = str(DATA_DIR / "appendix/represent_10CV_100k")
    
    if df is None:
        df = pkl.load(open(DATA_DIR / "training/Site_Observation_with_feature.pkl", 'rb'))
        df["time"] = pd.to_datetime(df["time"])
    
    if strategies is None:
        strategies = ['train', 'Sample', 'Site', 'Grid', 'Spatiotemporal_block']
    
    # Calculate global sparsity once
    df_with_sparsity = None
    if global_reference:
        print(f"Calculating global data sparsity (radius: {radius}km)...")
        df_with_sparsity = calculate_data_sparsity(
            df, reference_idx=df.index, output_idx=df.index,
            radius=radius, verbose=1
        )
        print("✓ Global sparsity calculation complete.")
    
    # Determine display names
    display_name1, display_name2 = name
    
    # Load metadata
    meta_json_path = os.path.join(project_path, 'meta_split.json')
    if not os.path.exists(meta_json_path):
        raise FileNotFoundError(f"meta_split.json not found in {project_path}")
    
    with open(meta_json_path, 'r') as f:
        metadata = json.load(f)
    
    configurations = metadata.get('configurations', [])
    
    # Create figure
    n_strategies = len(strategies)
    fig, axes = plt.subplots(1, n_strategies, figsize=figsize)
    if n_strategies == 1:
        axes = [axes]
    
    # Strategy display names
    strategy_display = {
        'train': 'Training',
        'Sample': 'Point-wise',
        'Site': 'Site-wise',
        'Grid': 'Region-wise',
        'Spatiotemporal_block': 'Spatiotemporal'
    }
    
    # Model colors
    colors = {model1: '#2E86AB', model2: '#A23B72'}
    
    for idx, strategy in enumerate(strategies):
        ax = axes[idx]
        
        # Find configuration
        config = None
        for c in configurations:
            if c['flag'] == strategy and c.get('sample_size') == sample_size:
                config = c
                break
        
        if config is None:
            ax.text(0.5, 0.5, f'No data for {strategy}', 
                   ha='center', va='center', transform=ax.transAxes)
            continue
        
        n_folds = len(config['folds'])
        fold_data = []
        
        print(f"\nProcessing {strategy} - {n_folds} folds...")
        
        for fold_idx in range(n_folds):
            fold_info = config['folds'][fold_idx]
            subdir = config['subdirectory']
            
            train_file = os.path.join(project_path, subdir, fold_info['train_file'])
            test_file = os.path.join(project_path, subdir, fold_info['test_file'])
            
            train_indices = np.load(train_file)
            test_indices = np.load(test_file)
            
            # Load predictions
            pred_file1 = os.path.join(project_path, model1, 
                                      f"{strategy}_size_{sample_size}_fold_{fold_idx+1}.npz")
            pred_file2 = os.path.join(project_path, model2, 
                                      f"{strategy}_size_{sample_size}_fold_{fold_idx+1}.npz")
            
            if not os.path.exists(pred_file1) or not os.path.exists(pred_file2):
                continue
            
            data1 = np.load(pred_file1)
            data2 = np.load(pred_file2)
            
            y_pred1 = np.ravel(data1['y_pred'])
            y_pred2 = np.ravel(data2['y_pred'])
            y_true = np.ravel(data1['y_true'])
            
            sparsity = df_with_sparsity.loc[test_indices, 'data_sparsity'].values
            sparsity = np.ravel(sparsity) * 10000
            
            min_len = min(len(sparsity), len(y_pred1), len(y_pred2), len(y_true))
            
            fold_data.append({
                'sparsity': sparsity[:min_len],
                'y_true1': y_true[:min_len],
                'y_pred1': y_pred1[:min_len],
                'y_true2': y_true[:min_len],
                'y_pred2': y_pred2[:min_len]
            })
        
        if len(fold_data) == 0:
            continue
        
        # Combine all folds
        all_sparsity = np.concatenate([fold['sparsity'] for fold in fold_data])
        
        # Determine bins
        sparsity_bins = np.linspace(all_sparsity.min(), all_sparsity.max(), n_bins + 1)
        bin_centers = (sparsity_bins[:-1] + sparsity_bins[1:]) / 2
        
        # Calculate R2 per bin per fold
        n_folds_used = len(fold_data)
        fold_bin_r2_1 = np.full((n_folds_used, n_bins), np.nan)
        fold_bin_r2_2 = np.full((n_folds_used, n_bins), np.nan)
        
        for fold_idx, fold in enumerate(fold_data):
            sparsity_fold = fold['sparsity']
            for bin_idx in range(n_bins):
                mask = (sparsity_fold >= sparsity_bins[bin_idx]) & \
                       (sparsity_fold < sparsity_bins[bin_idx + 1])
                if mask.sum() > 10:
                    try:
                        fold_bin_r2_1[fold_idx, bin_idx] = r2_score(fold['y_true1'][mask], fold['y_pred1'][mask])
                        fold_bin_r2_2[fold_idx, bin_idx] = r2_score(fold['y_true2'][mask], fold['y_pred2'][mask])
                    except:
                        pass
        
        # Calculate mean and std
        bin_r2_1 = np.nanmean(fold_bin_r2_1, axis=0)
        bin_r2_2 = np.nanmean(fold_bin_r2_2, axis=0)
        bin_stds1 = np.nanstd(fold_bin_r2_1, axis=0)
        bin_stds2 = np.nanstd(fold_bin_r2_2, axis=0)
        
        # Remove nan values
        valid_mask = ~(np.isnan(bin_r2_1) | np.isnan(bin_r2_2))
        bin_centers_valid = bin_centers[valid_mask]
        bin_r2_1 = bin_r2_1[valid_mask]
        bin_r2_2 = bin_r2_2[valid_mask]
        bin_stds1 = bin_stds1[valid_mask]
        bin_stds2 = bin_stds2[valid_mask]
        
        # Plot scatter points
        if visual_point_bin > 0:
            visual_bins = np.linspace(all_sparsity.min(), all_sparsity.max(), visual_point_bin + 1)
            visual_centers = (visual_bins[:-1] + visual_bins[1:]) / 2
            
            for fold in fold_data:
                sparsity_fold = fold['sparsity']
                scatter_r2_1, scatter_r2_2, scatter_density = [], [], []
                
                for bin_idx in range(visual_point_bin):
                    mask = (sparsity_fold >= visual_bins[bin_idx]) & \
                           (sparsity_fold < visual_bins[bin_idx + 1])
                    if mask.sum() > 10:
                        try:
                            scatter_r2_1.append(r2_score(fold['y_true1'][mask], fold['y_pred1'][mask]))
                            scatter_r2_2.append(r2_score(fold['y_true2'][mask], fold['y_pred2'][mask]))
                            scatter_density.append(visual_centers[bin_idx])
                        except:
                            pass
                
                if scatter_r2_1:
                    ax.scatter(scatter_density, scatter_r2_1, c=colors[model1], alpha=0.5, s=22, edgecolors='none')
                    ax.scatter(scatter_density, scatter_r2_2, c=colors[model2], alpha=0.5, s=22, edgecolors='none')
        
        # Plot trend lines
        ax.plot(bin_centers_valid, bin_r2_1, color=colors[model1], linewidth=3, alpha=0.9, zorder=5, label=display_name1)
        ax.plot(bin_centers_valid, bin_r2_2, color=colors[model2], linewidth=3, alpha=0.9, zorder=5, label=display_name2)
        
        # Uncertainty bands
        if uncertainty and n_folds_used > 1:
            ax.fill_between(bin_centers_valid, bin_r2_1 - bin_stds1, bin_r2_1 + bin_stds1,
                           color=colors[model1], alpha=0.2, zorder=1)
            ax.fill_between(bin_centers_valid, bin_r2_2 - bin_stds2, bin_r2_2 + bin_stds2,
                           color=colors[model2], alpha=0.2, zorder=1)
        
        ax.set_ylim(0, 1.0)
        
        if x_label:
            ax.set_xlabel('Data Density (×10⁴)', fontsize=font_size)
        if y_label and idx == 0:
            ax.set_ylabel('R² Score', fontsize=font_size)
        if show_title:
            ax.set_title(strategy_display.get(strategy, strategy), fontsize=font_size+1, fontweight='bold')
        if legend:
            ax.legend(loc='best', fontsize=font_size-2, framealpha=0.8)
        
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.tick_params(axis='both', labelsize=font_size-2)
        
        # Set tick locators
        from matplotlib.ticker import MaxNLocator, FuncFormatter
        if x_lim_number is not None:
            ax.xaxis.set_major_locator(MaxNLocator(nbins=x_lim_number-1, prune=None))
        if y_lim_number is not None:
            ax.yaxis.set_major_locator(MaxNLocator(nbins=y_lim_number-1, prune=None))
        
        ax.xaxis.set_major_formatter(FuncFormatter(lambda x, p: f'{x:.1f}'))
        ax.yaxis.set_major_formatter(FuncFormatter(lambda y, p: f'{y:.1f}'))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Figure saved to: {save_path}")
    
    return fig, axes
