"""
Figure 2: Model Performance Comparison
2x2 Grid Layout using GridSpec:
- Top-left: LightGBM (parameter vs error)
- Bottom-left: MLP (parameter vs error)
- Top-right: ResNet (epoch training curves)
- Bottom-right: Transformer (epoch training curves)

Standalone implementation - no external dependencies required.
"""

import os
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, Optional, List, Dict
from matplotlib.ticker import MaxNLocator, FuncFormatter
from matplotlib.patches import Rectangle
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter

# =============================================================================
# Local path constants
# =============================================================================
DATA_DIR = Path(__file__).parent.parent / "data"

# Strategy mapping for display names
STRATEGY_MAPPING = {
    'train': 'Training',
    'Sample': 'Time-wise',
    'Site': 'Site-wise',
    'Grid': 'Region-wise',
    'Spatiotemporal_block': 'Region&Time',
}

# Strategy colors (matching original OG_transformer)
STRATEGY_COLORS = {
    'train': 'grey',
    'Sample': '#2E86AB',
    'Site': '#A23B72',
    'Grid': '#2ca02c',
    'Time': '#d62728',
    'OrderTime': '#8c564b',
}


# =============================================================================
# Helper: Load model data
# =============================================================================

def _load_model_data(project_path: str, model_name: str) -> pd.DataFrame:
    """Load data for a model from CSV."""
    csv_file = os.path.join(project_path, f"{model_name}.csv")
    if os.path.exists(csv_file):
        return pd.read_csv(csv_file)
    return pd.DataFrame()


# =============================================================================
# Helper: Curve fitting (matching original implementation)
# =============================================================================

def _apply_fit(x_vals, y_vals, std_vals, fit_type='polynomial', degree=2, x_full_range=None):
    """Apply curve fitting to data (matching original OG_transformer implementation)."""
    x_fit = x_vals
    if x_full_range is None:
        x_full_range = x_vals
    
    # Calculate extended range
    x_range = x_full_range.max() - x_full_range.min()
    extend_ratio = 0.02
    x_min_ext = x_full_range.min() - x_range * extend_ratio
    x_max_ext = x_full_range.max() + x_range * extend_ratio
    
    if fit_type == 'peak_decay':
        # Peak decay: linear interpolation then smooth with Savitzky-Golay filter
        linear_interp = interp1d(x_fit, y_vals, kind='linear', fill_value='extrapolate')
        linear_interp_upper = interp1d(x_fit, y_vals + std_vals, kind='linear', fill_value='extrapolate')
        linear_interp_lower = interp1d(x_fit, y_vals - std_vals, kind='linear', fill_value='extrapolate')
        
        x_smooth = np.linspace(x_min_ext, x_max_ext, 100)
        y_line = linear_interp(x_smooth)
        y_upper = linear_interp_upper(x_smooth)
        y_lower = linear_interp_lower(x_smooth)
        
        # Savitzky-Golay filter
        window = min(31, len(x_smooth) if len(x_smooth) % 2 == 1 else len(x_smooth) - 1)
        polyorder = min(3, window - 1)
        y_smooth = savgol_filter(y_line, window_length=window, polyorder=polyorder)
        y_upper = savgol_filter(y_upper, window_length=window, polyorder=polyorder)
        y_lower = savgol_filter(y_lower, window_length=window, polyorder=polyorder)
    else:
        # Polynomial fit
        try:
            poly_coeffs = np.polyfit(x_fit, y_vals, degree)
            poly_func = np.poly1d(poly_coeffs)
            
            x_smooth = np.linspace(x_min_ext, x_max_ext, 100)
            y_smooth = poly_func(x_smooth)
            
            poly_upper = np.polyfit(x_fit, y_vals + std_vals, degree)
            poly_lower = np.polyfit(x_fit, y_vals - std_vals, degree)
            y_upper = np.poly1d(poly_upper)(x_smooth)
            y_lower = np.poly1d(poly_lower)(x_smooth)
        except:
            x_smooth = x_vals
            y_smooth = y_vals
            y_upper = y_vals + std_vals
            y_lower = y_vals - std_vals
    
    return x_smooth, y_smooth, y_upper, y_lower


# =============================================================================
# Helper: Plot strategy on axis (matching original style)
# =============================================================================

def _plot_strategy_on_axis(axis, df, strategy, x_col, y_col, 
                           fit_type='polynomial', degree=2,
                           confidence_band=True, show_points=True,
                           use_log_scale=False, exclude_first_n=0,
                           marker='o'):
    """Plot a single strategy on the given axis with original styling."""
    strategy_df = df[df['strategy'] == strategy]
    if len(strategy_df) == 0:
        return
    
    # Aggregate by x_col
    stats = strategy_df.groupby(x_col)[y_col].agg(['mean', 'std']).reset_index()
    stats.columns = ['x_value', 'mean', 'std']
    stats['std'] = stats['std'].fillna(0)
    stats = stats.sort_values('x_value')
    
    color = STRATEGY_COLORS.get(strategy, '#333333')
    label = STRATEGY_MAPPING.get(strategy, strategy)
    
    # Full range for curve extension
    x_full_range = stats['x_value'].values
    
    # Exclude first n points for fitting
    if exclude_first_n > 0 and len(stats) > exclude_first_n:
        stats_fit = stats.iloc[exclude_first_n:]
    else:
        stats_fit = stats
    
    x_vals = stats_fit['x_value'].values
    y_vals = stats_fit['mean'].values
    std_vals = stats_fit['std'].values
    
    # Apply log transform if needed
    if use_log_scale:
        x_vals_fit = np.log10(x_vals)
        x_full_fit = np.log10(x_full_range)
    else:
        x_vals_fit = x_vals
        x_full_fit = x_full_range
    
    # Plot based on fit type
    if fit_type in ['-', '--']:
        # Direct line plot
        axis.plot(stats['x_value'], stats['mean'], fit_type, 
                 color=color, label=label, linewidth=2)
        if confidence_band:
            axis.fill_between(stats['x_value'], stats['mean'] - stats['std'],
                            stats['mean'] + stats['std'], color=color, alpha=0.2)
    else:
        # Curve fitting
        x_smooth, y_smooth, y_upper, y_lower = _apply_fit(
            x_vals_fit, y_vals, std_vals, fit_type, degree, x_full_fit
        )
        
        # Transform back for log scale
        if use_log_scale:
            x_plot = 10**x_smooth
        else:
            x_plot = x_smooth
        
        axis.plot(x_plot, y_smooth, '-', color=color, label=label, linewidth=2)
        if confidence_band:
            axis.fill_between(x_plot, y_lower, y_upper, color=color, alpha=0.2)
        
        # Plot original points
        if show_points:
            axis.plot(stats['x_value'], stats['mean'], marker, 
                     color=color, markersize=4, alpha=0.6)


# =============================================================================
# Figure 2: Main Plotting Function (matching original OG_transformer style)
# =============================================================================

def plot_figure2(param_project_path="data/appendix/5CV_validation_vs_parameter",
                 epoch_project_path="data/appendix/5-Fold-Epoch",
                 figsize=(12, 10),
                 config=1,
                 save_path=None):
    """
    Create a 2x2 comparison figure of model performance.
    Matches original OG_transformer/plot/figure2.py styling.
    
    Parameters
    ----------
    param_project_path : str
        Path to parameter space analysis results
    epoch_project_path : str
        Path to epoch training curves results
    figsize : tuple
        Figure size (width, height)
    config : int
        Configuration (1=basic, 2=with confidence bands)
    save_path : str, optional
        Path to save the figure
    
    Returns
    -------
    fig : matplotlib.figure.Figure
    axes : dict
    """
    confidence_band = (config == 2)
    
    # Create figure and GridSpec (matching original)
    fig = plt.figure(figsize=figsize)
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.6)
    
    ax_lgb = fig.add_subplot(gs[0, 0])
    ax_mlp = fig.add_subplot(gs[1, 0])
    ax_resnet = fig.add_subplot(gs[0, 1])
    ax_transformer = fig.add_subplot(gs[1, 1])
    
    # =========================================================================
    # LightGBM (top-left) - Parameter vs R²
    # =========================================================================
    df_lgb = _load_model_data(param_project_path, 'lightgbm')
    if len(df_lgb) > 0:
        # Get parameter column
        param_cols = [c for c in df_lgb.columns if c not in 
                     ['model', 'parameter_number', 'split', 'strategy', 
                      'sample_size', 'fold', 'r2_score', 'fit_time']]
        if param_cols:
            param_col = param_cols[0]
            param_mapping = df_lgb.groupby(param_col)['parameter_number'].mean().to_dict()
            df_lgb['x_value'] = df_lgb[param_col].map(param_mapping)
            df_lgb = df_lgb.sort_values('x_value')
            
            # Plot left strategies (Sample)
            _plot_strategy_on_axis(ax_lgb, df_lgb, 'Sample', 'x_value', 'r2_score',
                                  fit_type='polynomial', degree=2,
                                  confidence_band=confidence_band, use_log_scale=True)
            
            # Create twin axis for right strategies
            ax_lgb2 = ax_lgb.twinx()
            _plot_strategy_on_axis(ax_lgb2, df_lgb, 'Site', 'x_value', 'r2_score',
                                  fit_type='polynomial', degree=2,
                                  confidence_band=confidence_band, use_log_scale=True)
            _plot_strategy_on_axis(ax_lgb2, df_lgb, 'Grid', 'x_value', 'r2_score',
                                  fit_type='polynomial', degree=2,
                                  confidence_band=confidence_band, use_log_scale=True)
            
            # Set limits (matching original)
            ax_lgb.set_ylim(0.55, 0.75)
            ax_lgb2.set_ylim(0.5, 0.625)
            ax_lgb.set_xscale('log')
            
            # Apply formatting (matching original)
            ax_lgb.set_ylabel('R² Score (Time-wise)', color='black')
            ax_lgb2.set_ylabel('Site-wise & Region-wise', color='black')
            ax_lgb.tick_params(axis='y', labelcolor='black')
            ax_lgb2.tick_params(axis='y', labelcolor='black')
            ax_lgb.yaxis.set_major_locator(MaxNLocator(nbins=6, prune=None))
            ax_lgb.yaxis.set_major_formatter(FuncFormatter(lambda x, p: f'{x:.2f}'))
            ax_lgb2.yaxis.set_major_locator(MaxNLocator(nbins=5, prune=None))
            ax_lgb2.yaxis.set_major_formatter(FuncFormatter(lambda x, p: f'{x:.2f}'))
            ax_lgb.grid(True, alpha=0.3)
            ax_lgb.margins(x=0)
    
    # =========================================================================
    # MLP (bottom-left) - Parameter vs R²
    # =========================================================================
    df_mlp = _load_model_data(param_project_path, 'mlp')
    if len(df_mlp) > 0:
        param_cols = [c for c in df_mlp.columns if c not in 
                     ['model', 'parameter_number', 'split', 'strategy', 
                      'sample_size', 'fold', 'r2_score', 'fit_time']]
        if param_cols:
            param_col = param_cols[0]
            param_mapping = df_mlp.groupby(param_col)['parameter_number'].mean().to_dict()
            df_mlp['x_value'] = df_mlp[param_col].map(param_mapping)
            df_mlp = df_mlp.sort_values('x_value')
            
            _plot_strategy_on_axis(ax_mlp, df_mlp, 'Sample', 'x_value', 'r2_score',
                                  fit_type='polynomial', degree=2,
                                  confidence_band=confidence_band, use_log_scale=True)
            
            ax_mlp2 = ax_mlp.twinx()
            _plot_strategy_on_axis(ax_mlp2, df_mlp, 'Site', 'x_value', 'r2_score',
                                  fit_type='polynomial', degree=2,
                                  confidence_band=confidence_band, use_log_scale=True)
            _plot_strategy_on_axis(ax_mlp2, df_mlp, 'Grid', 'x_value', 'r2_score',
                                  fit_type='polynomial', degree=2,
                                  confidence_band=confidence_band, use_log_scale=True)
            
            ax_mlp.set_ylim(0.5, 0.625)
            ax_mlp2.set_ylim(0.5, 0.625)
            ax_mlp.set_xscale('log')
            ax_mlp.set_xlabel('Parameter Count')
            ax_mlp.set_ylabel('R² Score (Time-wise)', color='black')
            ax_mlp2.set_ylabel('Site-wise & Region-wise', color='black')
            ax_mlp.tick_params(axis='y', labelcolor='black')
            ax_mlp2.tick_params(axis='y', labelcolor='black')
            ax_mlp.yaxis.set_major_locator(MaxNLocator(nbins=6, prune=None))
            ax_mlp.yaxis.set_major_formatter(FuncFormatter(lambda x, p: f'{x:.2f}'))
            ax_mlp2.yaxis.set_major_locator(MaxNLocator(nbins=5, prune=None))
            ax_mlp2.yaxis.set_major_formatter(FuncFormatter(lambda x, p: f'{x:.2f}'))
            ax_mlp.grid(True, alpha=0.3)
            ax_mlp.margins(x=0)
    
    # =========================================================================
    # ResNet (top-right) - Epoch vs R²
    # =========================================================================
    df_resnet = _load_model_data(epoch_project_path, 'resnet')
    if len(df_resnet) > 0:
        _plot_strategy_on_axis(ax_resnet, df_resnet, 'Sample', 'epoch', 'val_r2',
                              fit_type='polynomial', degree=2,
                              confidence_band=confidence_band, exclude_first_n=1)
        
        ax_resnet2 = ax_resnet.twinx()
        _plot_strategy_on_axis(ax_resnet2, df_resnet, 'Site', 'epoch', 'val_r2',
                              fit_type='peak_decay',
                              confidence_band=confidence_band, exclude_first_n=1)
        _plot_strategy_on_axis(ax_resnet2, df_resnet, 'Grid', 'epoch', 'val_r2',
                              fit_type='peak_decay',
                              confidence_band=confidence_band, exclude_first_n=1)
        
        ax_resnet.set_ylim(0.46, 0.62)
        ax_resnet2.set_ylim(0.46, 0.62)
        ax_resnet.set_ylabel('R² Score (Time-wise)', color='black')
        ax_resnet2.set_ylabel('Site-wise & Region-wise', color='black')
        ax_resnet.tick_params(axis='y', labelcolor='black')
        ax_resnet2.tick_params(axis='y', labelcolor='black')
        ax_resnet.yaxis.set_major_locator(MaxNLocator(nbins=6, prune=None))
        ax_resnet.yaxis.set_major_formatter(FuncFormatter(lambda x, p: f'{x:.2f}'))
        ax_resnet2.yaxis.set_major_locator(MaxNLocator(nbins=6, prune=None))
        ax_resnet2.yaxis.set_major_formatter(FuncFormatter(lambda x, p: f'{x:.2f}'))
        ax_resnet.grid(True, alpha=0.3)
        ax_resnet.margins(x=0)
    
    # =========================================================================
    # Transformer (bottom-right) - Epoch vs R²
    # =========================================================================
    df_trans = _load_model_data(epoch_project_path, 'transformer')
    if len(df_trans) > 0:
        _plot_strategy_on_axis(ax_transformer, df_trans, 'Sample', 'epoch', 'val_r2',
                              fit_type='polynomial', degree=2,
                              confidence_band=confidence_band, exclude_first_n=1)
        
        ax_transformer2 = ax_transformer.twinx()
        _plot_strategy_on_axis(ax_transformer2, df_trans, 'Site', 'epoch', 'val_r2',
                              fit_type='polynomial', degree=2,
                              confidence_band=confidence_band, exclude_first_n=1)
        _plot_strategy_on_axis(ax_transformer2, df_trans, 'Grid', 'epoch', 'val_r2',
                              fit_type='polynomial', degree=2,
                              confidence_band=confidence_band, exclude_first_n=1)
        
        ax_transformer.set_ylim(0.5, 0.65)
        ax_transformer2.set_ylim(0.5, 0.65)
        ax_transformer.set_xlabel('Epoch')
        ax_transformer.set_ylabel('R² Score (Time-wise)', color='black')
        ax_transformer2.set_ylabel('Site-wise & Region-wise', color='black')
        ax_transformer.tick_params(axis='y', labelcolor='black')
        ax_transformer2.tick_params(axis='y', labelcolor='black')
        ax_transformer.yaxis.set_major_locator(MaxNLocator(nbins=6, prune=None))
        ax_transformer.yaxis.set_major_formatter(FuncFormatter(lambda x, p: f'{x:.2f}'))
        ax_transformer2.yaxis.set_major_locator(MaxNLocator(nbins=6, prune=None))
        ax_transformer2.yaxis.set_major_formatter(FuncFormatter(lambda x, p: f'{x:.2f}'))
        ax_transformer.grid(True, alpha=0.3)
        ax_transformer.margins(x=0)
        
        # Combined legend for transformer (show_legend=True in original)
        lines1, labels1 = ax_transformer.get_legend_handles_labels()
        lines2, labels2 = ax_transformer2.get_legend_handles_labels()
        ax_transformer.legend(lines1 + lines2, labels1 + labels2, loc='lower right')
    
    plt.tight_layout()
    
    if save_path:
        if save_path.lower().endswith('.svg'):
            fig.savefig(save_path, format='svg', bbox_inches='tight')
        else:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Figure 2 saved to: {save_path}")
    
    return fig, {'lightgbm': ax_lgb, 'mlp': ax_mlp, 'resnet': ax_resnet, 'transformer': ax_transformer}


# =============================================================================
# Figure 2c: Global and Regional Maps (matching original)
# =============================================================================

def plot_global_and_regions(ds, regions, cmap='viridis', vmin=20, vmax=70, 
                            variable='o3_ml', figsize=(14, 16), save_path=None,
                            extend='both', show_labels=False, region_labels=None,
                            box_color='black', box_linewidth=2, R2_information=None,
                            main_title_font=16, sub_title_font=14, text_color='black', alpha_text=1):
    """
    Create a figure with global map (top) and regional maps (bottom).
    Matches original OG_transformer/plot/figure2.py styling.
    
    Parameters
    ----------
    ds : xarray.Dataset or xarray.DataArray
        Input dataset
    regions : list of lists
        List of regions, each as [lat_min, lon_min, lat_max, lon_max]
    cmap : str
        Colormap name
    vmin, vmax : float
        Color scale limits
    variable : str
        Variable name to plot
    figsize : tuple
        Figure size
    save_path : str, optional
        Path to save figure
    extend : str
        Colorbar extension
    show_labels : bool
        Whether to show axis labels
    region_labels : list of str, optional
        Labels for regions (default: A, B, C, D)
    box_color : str
        Color of region boxes
    box_linewidth : float
        Line width of boxes
    R2_information : list, optional
        [df, train_ind, test_ind] for R2 calculation
    main_title_font : int
        Font size for R2 text
    sub_title_font : int
        Font size for region labels
    text_color : str
        Color for text
    alpha_text : float
        Text transparency
    
    Returns
    -------
    fig, axes : figure and axes dictionary
    """
    # Handle Dataset vs DataArray
    if hasattr(ds, 'data_vars'):
        data = ds[variable]
    else:
        data = ds
    
    # Default region labels
    if region_labels is None:
        region_labels = [chr(65 + i) for i in range(len(regions))]
    
    # Calculate R2 for each region
    region_r2_scores = {}
    overall_r2 = None
    if R2_information is not None:
        from sklearn.metrics import r2_score
        df, train_ind, test_ind = R2_information
        test_df = df.loc[test_ind]
        
        if len(test_df) > 0 and 'predicted' in test_df.columns and 'Ozone' in test_df.columns:
            overall_r2 = r2_score(test_df['Ozone'], test_df['predicted'])
            if overall_r2 <= 0:
                overall_r2 = 0.0
        
        for i, region in enumerate(regions):
            lat_min, lon_min, lat_max, lon_max = region
            region_mask = (
                (test_df['latitude'] >= lat_min) & (test_df['latitude'] <= lat_max) &
                (test_df['longitude'] >= lon_min) & (test_df['longitude'] <= lon_max)
            )
            region_test_df = test_df[region_mask]
            
            if len(region_test_df) > 0:
                r2 = r2_score(region_test_df['Ozone'], region_test_df['predicted'])
                region_r2_scores[region_labels[i]] = max(0, r2)
    
    # Create figure with 4x4 GridSpec (matching original)
    fig = plt.figure(figsize=figsize)
    gs = gridspec.GridSpec(4, 4, figure=fig, 
                          width_ratios=[1, 1, 1, 0.15],
                          hspace=0.4, wspace=0.3)
    
    # Global map (rows 0-1, columns 0-2)
    ax_global = fig.add_subplot(gs[0:2, 0:3])
    
    # Handle extra dimensions
    global_data = data
    while global_data.ndim > 2:
        dims = list(global_data.dims)
        extra_dims = [d for d in dims if d not in ['latitude', 'longitude', 'lat', 'lon']]
        if extra_dims:
            global_data = global_data.isel({extra_dims[0]: 0})
        else:
            break
    
    # Plot global map
    lons = global_data.longitude.values
    lats = global_data.latitude.values
    values = global_data.values
    
    im = ax_global.pcolormesh(lons, lats, values, cmap=cmap, vmin=vmin, vmax=vmax, shading='auto')
    ax_global.set_xlabel('Longitude', fontsize=12)
    ax_global.set_ylabel('Latitude', fontsize=12)
    
    # Overall R2 (matching original position and style)
    if overall_r2 is not None:
        ax_global.text(0.02, 0.98, f'R² = {overall_r2:.2f}', 
                      transform=ax_global.transAxes,
                      fontsize=main_title_font, fontweight='bold',
                      verticalalignment='top', horizontalalignment='left',
                      color=text_color, alpha=alpha_text)
    
    # Region boxes (matching original style)
    for i, (region, label) in enumerate(zip(regions, region_labels)):
        lat_min, lon_min, lat_max, lon_max = region
        width = lon_max - lon_min
        height = lat_max - lat_min
        
        rect = Rectangle((lon_min, lat_min), width, height,
                         linewidth=box_linewidth, edgecolor=box_color,
                         facecolor='none', linestyle='--', zorder=10, alpha=alpha_text)
        ax_global.add_patch(rect)
        
        # Label inside box at top-left
        offset_x = width * 0.05
        offset_y = height * 0.05
        ax_global.text(lon_min + offset_x, lat_max - offset_y, label,
                      color=box_color, fontsize=sub_title_font, fontweight='bold',
                      verticalalignment='top', horizontalalignment='left',
                      zorder=11, alpha=alpha_text)
    
    # Colorbar (matching original)
    ax_cbar = fig.add_subplot(gs[0:2, 3])
    cbar = plt.colorbar(im, cax=ax_cbar, extend=extend)
    cbar.set_label("Ozone (ppb)", fontsize=12)
    
    # Regional maps (rows 2-3, columns 0-2)
    gs_regions = gridspec.GridSpecFromSubplotSpec(2, 2, subplot_spec=gs[2:4, 0:3],
                                                  wspace=0.3, hspace=0.3)
    
    axes_regions = []
    for i, region in enumerate(regions):
        row, col = i // 2, i % 2
        ax = fig.add_subplot(gs_regions[row, col])
        axes_regions.append(ax)
        
        lat_min, lon_min, lat_max, lon_max = region
        region_data = global_data.sel(latitude=slice(lat_min, lat_max), 
                                      longitude=slice(lon_min, lon_max))
        
        if region_data.size > 0:
            ax.pcolormesh(region_data.longitude, region_data.latitude, region_data.values,
                         cmap=cmap, vmin=vmin, vmax=vmax, shading='auto')
        
        # Label with R2 (matching original style)
        label = region_labels[i]
        if label in region_r2_scores:
            label_text = f'{label} (R²={region_r2_scores[label]:.2f})'
        else:
            label_text = label
        
        ax.text(0.05, 0.95, label_text, transform=ax.transAxes,
               fontsize=main_title_font, fontweight='bold',
               verticalalignment='top', horizontalalignment='left',
               color=text_color, alpha=alpha_text)
        
        if show_labels:
            ax.set_xlabel('Longitude', fontsize=10)
            ax.set_ylabel('Latitude', fontsize=10)
    
    if save_path:
        if save_path.lower().endswith('.svg'):
            fig.savefig(save_path, format='svg', bbox_inches='tight')
        else:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Figure saved to: {save_path}")
    
    return fig, {'global': ax_global, 'regions': axes_regions}


# Aliases (matching original)
plot_complexity_epochs_vs_r2 = plot_figure2
plot_spatial_prediction_regions = plot_global_and_regions
