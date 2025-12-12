"""
Figure 3: OG Ablation Study Results

This module provides visualization comparing model performance across different
validation strategies (Site-based vs Grid-based) for:
- Deep Learning models alone (e.g., MLP, ResNet, Transformer)
- Machine Learning models alone (e.g., LightGBM, XGBoost, CatBoost)
- Hybrid models (DL + ML combination for OG component)

Ported from OG_transformer.appendix.plot_og_result
"""

import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.path as mpath
from matplotlib.patches import Rectangle
from matplotlib.ticker import MultipleLocator, MaxNLocator, FormatStrFormatter
from matplotlib.colors import BoundaryNorm, LinearSegmentedColormap, Normalize
from matplotlib.path import Path
from matplotlib.cm import ScalarMappable
from pathlib import Path as FilePath
from typing import Tuple, Optional, List, Dict, Union

# =============================================================================
# Local path constants
# =============================================================================
DATA_DIR = FilePath(__file__).parent.parent / "data"


# =============================================================================
# Helper functions
# =============================================================================

def _create_hexagram_marker():
    """
    Create a hexagram (six-pointed star, Star of David) marker.
    
    Returns
    -------
    matplotlib.path.Path
        Custom marker path for hexagram (6-pointed star)
    """
    return mpath.Path.unit_regular_star(6)


def _get_colormap(cmap):
    """
    Get colormap from string name or return existing colormap object.
    
    Supports a custom 'gray_blue' colormap for elegant gray-to-blue gradient.
    """
    if isinstance(cmap, str):
        if cmap == 'gray_blue':
            # Create custom gray-to-blue gradient
            colors = [
                (0.85, 0.85, 0.85),  # Light gray
                (0.65, 0.65, 0.65),  # Medium gray
                (0.7, 0.8, 0.95),    # Light blue
                (0.3, 0.5, 0.8),     # Medium blue
                (0.1, 0.3, 0.6)      # Deep blue
            ]
            return LinearSegmentedColormap.from_list('gray_blue', colors)
        else:
            return plt.cm.get_cmap(cmap)
    else:
        return cmap


def _standardize_model_name(model_name):
    """
    Standardize model names for display (case-insensitive matching).
    
    Examples:
        'biglightgbm', 'BigLightGBM', 'LIGHTGBM' -> 'LightGBM'
        'bigxgboost', 'BigXGBoost', 'XGBOOST' -> 'XGBoost'
        'mlp_early', 'MLP_EARLY' -> 'MLP'
    """
    standardization_map = {
        'lightgbm': 'LightGBM',
        'biglightgbm': 'LightGBM',
        'light_gbm': 'LightGBM',
        'lgbm': 'LightGBM',
        'xgboost': 'XGBoost',
        'bigxgboost': 'XGBoost',
        'xgb': 'XGBoost',
        'catboost': 'CatBoost',
        'bigcatboost': 'CatBoost',
        'cat_boost': 'CatBoost',
        'catb': 'CatBoost',
        'mlp': 'MLP',
        'mlp_early': 'MLP',
        'resnet': 'ResNet',
        'resnet_early': 'ResNet',
        'transformer': 'Transformer',
        'transformer_early': 'Transformer',
        'randomforest': 'RandomForest',
        'rf': 'RandomForest',
        'gbm': 'GBM',
        'adaboost': 'AdaBoost',
        'svm': 'SVM',
        'linear': 'Linear',
        'ridge': 'Ridge',
        'lasso': 'Lasso',
    }
    
    model_lower = model_name.lower().strip()
    
    if model_lower in standardization_map:
        return standardization_map[model_lower]
    
    return model_name.upper()


def _standardize_metric_name(metric_name):
    """
    Standardize metric/validation strategy names for display.
    
    Examples:
        'site', 'Site', 'SITE' -> 'Site R²'
        'grid', 'Grid', 'region' -> 'Region R²'
        'spatiotemporal', 'Spatiotemporal_block' -> 'Spatiotemporal R²'
    """
    standardization_map = {
        'site': 'Site',
        'grid': 'Region',
        'region': 'Region',
        'sample': 'Sample',
        'time': 'Time',
        'ordertime': 'OrderTime',
        'hours': 'Hours',
        'spatiotemporal': 'Spatiotemporal',
        'spatiotemporal_block': 'Spatiotemporal',
        'spatio_temporal': 'Spatiotemporal',
        'spatio-temporal': 'Spatiotemporal',
        'train': 'Train',
        'cv': 'CV',
        'random': 'Random',
    }
    
    metric_lower = metric_name.lower().strip()
    
    if metric_lower in standardization_map:
        return f'{standardization_map[metric_lower]} R²'
    
    return f'{metric_name.capitalize()} R²'


# =============================================================================
# Data loading
# =============================================================================

def load_og_results(project_path, metrics=["Site", "Grid"], sufficiency=None):
    """
    Load results from OG ablation experiment.
    
    Parameters
    ----------
    project_path : str
        Path to the results directory
    metrics : list of str
        List of validation strategies to load (e.g., ["Site", "Grid"])
    sufficiency : int or list of int, optional
        Sufficiency value(s) to filter by.
    
    Returns
    -------
    dict
        Dictionary with structure: {metric: {model: {'r2_mean': float, 'r2_std': float}}}
    """
    results = {metric: {} for metric in metrics}
    
    # Get all CSV files
    csv_pattern = os.path.join(project_path, "*.csv")
    csv_files = glob.glob(csv_pattern)
    
    # Skip meta files
    csv_files = [f for f in csv_files if 'meta' not in os.path.basename(f).lower()]
    
    # Convert sufficiency to list for easier handling
    if sufficiency is not None:
        if not isinstance(sufficiency, (list, tuple)):
            sufficiency = [sufficiency]
    
    for csv_file in csv_files:
        basename = os.path.basename(csv_file)
        model_name = basename.replace('.csv', '')
        
        try:
            df = pd.read_csv(csv_file)
            
            # Filter by sufficiency if specified
            if sufficiency is not None and 'sample_size' in df.columns:
                df = df[df['sample_size'].isin(sufficiency)]
            
            # Process each metric (validation strategy)
            for metric in metrics:
                # Filter by validation column
                if 'validation' in df.columns:
                    metric_df = df[df['validation'] == metric]
                elif 'strategy' in df.columns:
                    metric_df = df[df['strategy'] == metric]
                else:
                    continue
                
                # Calculate mean and std of R2
                if 'r2_score' in metric_df.columns:
                    r2_col = 'r2_score'
                elif 'r2' in metric_df.columns:
                    r2_col = 'r2'
                elif 'R2' in metric_df.columns:
                    r2_col = 'R2'
                else:
                    continue
                
                if len(metric_df) > 0:
                    results[metric][model_name] = {
                        'r2_mean': metric_df[r2_col].mean(),
                        'r2_std': metric_df[r2_col].std()
                    }
        except Exception as e:
            print(f"Warning: Error loading {csv_file}: {e}")
            continue
    
    return results


# =============================================================================
# Main plotting function
# =============================================================================

def plot_og_ablation(project_path, dl_group, ml_group, metrics=["Site", "Grid"], 
                     figsize=None, save_path=None, x_lim=None, y_lim=None, early_stop=False,
                     x_locator=None, y_locator=None, n_folds=10, color_metric=None, color_lim=None,
                     color_boundaries=None, cmap='Blues', bold=False, sufficiency=None):
    """
    Plot OG ablation study results comparing Site vs Grid validation.
    
    Creates an m×n grid of subplots (m=len(dl_group), n=len(ml_group)), where each subplot shows:
    - DL model alone (blue downward triangle '▼')
    - ML model alone (purple upward triangle '▲')
    - OG hybrid model (orange hexagram '✡' - Star of David)
    
    Parameters
    ----------
    project_path : str
        Path to the results directory containing CSV files
    dl_group : list of str
        List of deep learning model names (e.g., ['mlp', 'resnet', 'transformer'])
    ml_group : list of str
        List of machine learning model names (e.g., ['biglightgbm', 'bigxgboost', 'bigcatboost'])
    metrics : list of str, default=["Site", "Grid"]
        Validation strategies to compare (x-axis: first, y-axis: second)
    figsize : tuple, optional
        Figure size (width, height)
    save_path : str, optional
        Path to save the figure
    x_lim, y_lim : tuple, optional
        Axis limits as (min, max)
    early_stop : bool, default=False
        If True, use early stopping versions of DL models
    x_locator, y_locator : int or float, optional
        Axis tick control
    n_folds : int, default=10
        Number of CV folds for error bar calculation
    color_metric : str, optional
        Third metric for color-coding points
    color_lim : tuple, optional
        Color scale limits
    color_boundaries : list, optional
        Non-uniform color boundaries for colorbar
    cmap : str, default='Blues'
        Colormap name
    bold : bool, default=False
        If True, use bold font weight
    sufficiency : int or list of int, optional
        Sufficiency value(s) to filter by
    
    Returns
    -------
    fig, axes : matplotlib figure and axes objects
    
    Example
    -------
    >>> fig, axes = plot_og_ablation(
    ...     project_path="data/appendix/10-Fold",
    ...     dl_group=['mlp', 'resnet', 'transformer'],
    ...     ml_group=['biglightgbm', 'bigxgboost', 'bigcatboost'],
    ...     metrics=["Site", "Grid"],
    ...     early_stop=True,
    ...     x_lim=[0.57, 0.61],
    ...     y_lim=[0.5, 0.61],
    ...     color_metric="Spatiotemporal_block",
    ...     color_boundaries=[0.45, 0.46, 0.47, 0.48, 0.49, 0.50, 0.510, 0.520],
    ...     bold=True,
    ...     save_path="figures/figure3.svg"
    ... )
    """
    # Set global font sizes
    plt.rc('font', size=12)
    plt.rc('axes', titlesize=13)
    plt.rc('axes', labelsize=12)
    plt.rc('xtick', labelsize=10)
    plt.rc('ytick', labelsize=10)
    plt.rc('legend', fontsize=10)
    
    # Load results for x/y metrics
    results = load_og_results(project_path, metrics, sufficiency=sufficiency)
    
    # Load color metric if specified
    color_results = None
    if color_metric:
        color_results = load_og_results(project_path, [color_metric], sufficiency=sufficiency)
        print(f"✓ Loaded color metric: {color_metric}")
    
    # Print sufficiency info
    if sufficiency is not None:
        if isinstance(sufficiency, (list, tuple)):
            print(f"✓ Filtered by sufficiency: {sufficiency}")
        else:
            print(f"✓ Filtered by sufficiency: {sufficiency}")
    
    print(f"✓ Loaded results for {len(results[metrics[0]])} models")
    print(f"  Available models: {list(results[metrics[0]].keys())[:5]}...")
    
    # Determine figure layout
    n_rows = len(dl_group)
    n_cols = len(ml_group)
    
    if figsize is None:
        figsize = (5 * n_cols, 4 * n_rows)
    
    # Create figure with gridspec
    fig = plt.figure(figsize=figsize)
    if color_metric:
        gs = gridspec.GridSpec(n_rows, n_cols, figure=fig, wspace=0.35, hspace=0.35, right=0.92)
    else:
        gs = gridspec.GridSpec(n_rows, n_cols, figure=fig, wspace=0.35, hspace=0.35)
    
    # Color schemes
    colors = {
        'dl': '#2E86AB',      # Blue for DL models
        'ml': '#A23B72',      # Purple for ML models  
        'hybrid': '#F18F01'   # Orange for hybrid models
    }
    
    # Markers
    markers = {
        'dl': 'v',                      # Down triangle for DL
        'ml': '^',                      # Up triangle for ML
        'hybrid': _create_hexagram_marker()  # Hexagram for HYBRID
    }
    sizes = {
        'dl': 100,
        'ml': 100,
        'hybrid': 140
    }
    
    # Pre-calculate color limits and normalization
    vmin, vmax = None, None
    norm = None
    colormap = _get_colormap(cmap) if color_metric else None
    
    if color_metric:
        if color_boundaries is not None:
            boundaries = sorted(color_boundaries)
            vmin, vmax = boundaries[0], boundaries[-1]
            norm = BoundaryNorm(boundaries, colormap.N, clip=True)
            print(f"✓ Using non-uniform color scale with {len(boundaries)} boundaries")
            print(f"  Range: {vmin:.3f} - {vmax:.3f}")
        elif color_lim is None:
            all_color_values = []
            for dl_model in dl_group:
                for ml_model in ml_group:
                    dl_key = f"{dl_model.lower()}_early" if early_stop else dl_model.lower()
                    ml_key = ml_model.lower()
                    hybrid_key = f"{dl_model.lower()}+{ml_model.lower()}"
                    
                    for key in [dl_key, ml_key, hybrid_key]:
                        if key in color_results[color_metric]:
                            val = color_results[color_metric][key].get('r2_mean')
                            if val is not None:
                                all_color_values.append(val)
            
            if all_color_values:
                vmin, vmax = min(all_color_values), max(all_color_values)
            else:
                vmin, vmax = 0, 1
            norm = Normalize(vmin=vmin, vmax=vmax)
        else:
            vmin, vmax = color_lim
            norm = Normalize(vmin=vmin, vmax=vmax)
    
    axes = []
    
    # Create subplot for each DL × ML combination
    for row_idx, dl_model in enumerate(dl_group):
        row_axes = []
        
        for col_idx, ml_model in enumerate(ml_group):
            ax = fig.add_subplot(gs[row_idx, col_idx])
            row_axes.append(ax)
            
            # Get model keys
            if early_stop:
                dl_key = f"{dl_model.lower()}_early"
            else:
                dl_key = dl_model.lower()
            
            ml_key = ml_model.lower()
            hybrid_key = f"{dl_model.lower()}+{ml_model.lower()}"
            
            # Alternative hybrid key formats
            if hybrid_key not in results[metrics[0]]:
                alternative_keys = [
                    f"{dl_model.lower()}_{ml_model.lower()}",
                    f"{dl_model.lower()}+big{ml_model.lower()}",
                    f"big{dl_model.lower()}+{ml_model.lower()}",
                ]
                for alt_key in alternative_keys:
                    if alt_key in results[metrics[0]]:
                        hybrid_key = alt_key
                        break
            
            # Extract data for x-axis (first metric)
            dl_x = results[metrics[0]].get(dl_key, {}).get('r2_mean', None)
            ml_x = results[metrics[0]].get(ml_key, {}).get('r2_mean', None)
            hybrid_x = results[metrics[0]].get(hybrid_key, {}).get('r2_mean', None)
            
            dl_x_std = results[metrics[0]].get(dl_key, {}).get('r2_std', 0)
            ml_x_std = results[metrics[0]].get(ml_key, {}).get('r2_std', 0)
            hybrid_x_std = results[metrics[0]].get(hybrid_key, {}).get('r2_std', 0)
            
            # Extract data for y-axis (second metric)
            dl_y = results[metrics[1]].get(dl_key, {}).get('r2_mean', None)
            ml_y = results[metrics[1]].get(ml_key, {}).get('r2_mean', None)
            hybrid_y = results[metrics[1]].get(hybrid_key, {}).get('r2_mean', None)
            
            dl_y_std = results[metrics[1]].get(dl_key, {}).get('r2_std', 0)
            ml_y_std = results[metrics[1]].get(ml_key, {}).get('r2_std', 0)
            hybrid_y_std = results[metrics[1]].get(hybrid_key, {}).get('r2_std', 0)
            
            # Calculate standard error of mean (SEM)
            sem_factor = 1.0 / np.sqrt(n_folds)
            
            # Extract color data if color_metric is specified
            dl_color = ml_color = hybrid_color = None
            if color_results:
                dl_color = color_results[color_metric].get(dl_key, {}).get('r2_mean', None)
                ml_color = color_results[color_metric].get(ml_key, {}).get('r2_mean', None)
                hybrid_color = color_results[color_metric].get(hybrid_key, {}).get('r2_mean', None)
            
            # Plot DL model
            if dl_x is not None and dl_y is not None:
                if color_metric and dl_color is not None:
                    point_color = colormap(norm(dl_color))
                    ax.scatter(dl_x, dl_y, s=sizes['dl'], marker=markers['dl'],
                              c=[point_color], edgecolors='black', linewidths=0.5,
                              alpha=0.9, label='LV', zorder=3)
                    ax.errorbar(dl_x, dl_y,
                               xerr=dl_x_std * sem_factor, yerr=dl_y_std * sem_factor,
                               fmt='none', ecolor='gray', capsize=3, alpha=0.3, zorder=2)
                else:
                    ax.errorbar(
                        dl_x, dl_y,
                        xerr=dl_x_std * sem_factor, yerr=dl_y_std * sem_factor,
                        fmt=markers['dl'], color=colors['dl'],
                        markersize=np.sqrt(sizes['dl']), capsize=3,
                        alpha=0.7, label='LV', zorder=3
                    )
            
            # Plot ML model
            if ml_x is not None and ml_y is not None:
                if color_metric and ml_color is not None:
                    point_color = colormap(norm(ml_color))
                    ax.scatter(ml_x, ml_y, s=sizes['ml'], marker=markers['ml'],
                              c=[point_color], edgecolors='black', linewidths=0.5,
                              alpha=0.9, label='HV', zorder=4)
                    ax.errorbar(ml_x, ml_y,
                               xerr=ml_x_std * sem_factor, yerr=ml_y_std * sem_factor,
                               fmt='none', ecolor='gray', capsize=3, alpha=0.3, zorder=2)
                else:
                    ax.errorbar(
                        ml_x, ml_y,
                        xerr=ml_x_std * sem_factor, yerr=ml_y_std * sem_factor,
                        fmt=markers['ml'], color=colors['ml'],
                        markersize=np.sqrt(sizes['ml']), capsize=3,
                        alpha=0.7, label='HV', zorder=4
                    )
            
            # Plot Hybrid model
            if hybrid_x is not None and hybrid_y is not None:
                is_composite = isinstance(markers['hybrid'], tuple)
                
                if color_metric and hybrid_color is not None:
                    point_color = colormap(norm(hybrid_color))
                    
                    if is_composite:
                        ax.scatter(hybrid_x, hybrid_y, s=sizes['hybrid'], marker=markers['hybrid'][0],
                                  c=[point_color], edgecolors='black', linewidths=0.8,
                                  alpha=0.9, label='OG', zorder=5)
                        y_offset = (ax.get_ylim()[1] - ax.get_ylim()[0]) * 0.005
                        ax.scatter(hybrid_x, hybrid_y + y_offset, s=sizes['hybrid']*0.7, marker=markers['hybrid'][1],
                                  facecolors='none', edgecolors="white", linewidths=1.5,
                                  alpha=1, zorder=6)
                    else:
                        ax.scatter(hybrid_x, hybrid_y, s=sizes['hybrid'], marker=markers['hybrid'],
                                  c=[point_color], edgecolors='black', linewidths=0.8,
                                  alpha=0.9, label='OG', zorder=5)
                    
                    ax.errorbar(hybrid_x, hybrid_y,
                               xerr=hybrid_x_std * sem_factor, yerr=hybrid_y_std * sem_factor,
                               fmt='none', ecolor='gray', capsize=3, alpha=0.3, zorder=2)
                else:
                    if is_composite:
                        ax.scatter(hybrid_x, hybrid_y, s=sizes['hybrid'], marker=markers['hybrid'][0],
                                  color=colors['hybrid'], edgecolors='black', linewidths=0.8,
                                  alpha=0.9, label='OG', zorder=5)
                        y_offset = (ax.get_ylim()[1] - ax.get_ylim()[0]) * 0.005
                        ax.scatter(hybrid_x, hybrid_y + y_offset, s=sizes['hybrid']*0.7, marker=markers['hybrid'][1],
                                  facecolors='none', edgecolors='white', linewidths=2,
                                  alpha=1, zorder=6)
                        ax.errorbar(hybrid_x, hybrid_y,
                                   xerr=hybrid_x_std * sem_factor, yerr=hybrid_y_std * sem_factor,
                                   fmt='none', ecolor='gray', capsize=3, alpha=0.3, zorder=2)
                    else:
                        ax.errorbar(hybrid_x, hybrid_y,
                                   xerr=hybrid_x_std * sem_factor, yerr=hybrid_y_std * sem_factor,
                                   fmt=markers['hybrid'], color=colors['hybrid'],
                                   markersize=np.sqrt(sizes['hybrid']), capsize=3,
                                   alpha=0.9, label='OG', zorder=5, linewidth=1.5)
                
                # Draw arrow from DL to Hybrid (showing improvement)
                if dl_x is not None and dl_y is not None:
                    ax.annotate(
                        '', xy=(hybrid_x, hybrid_y),
                        xytext=(dl_x, dl_y),
                        arrowprops=dict(
                            arrowstyle='->', color="purple",
                            lw=2.2, alpha=0.4
                        ),
                        zorder=2
                    )
            
            # Add diagonal reference line (x = y)
            if dl_x and dl_y and ml_x and ml_y:
                lims = [
                    min(dl_x, ml_x, dl_y, ml_y) - 0.05,
                    max(dl_x, ml_x, dl_y, ml_y) + 0.05
                ]
                ax.plot(lims, lims, '--', color='gray', alpha=0.4, lw=1, zorder=1)
            
            # Formatting
            x_label = _standardize_metric_name(metrics[0]) if row_idx == n_rows - 1 else ''
            y_label = _standardize_metric_name(metrics[1]) if col_idx == 0 else ''
            fontweight = 'bold' if bold else 'normal'
            ax.set_xlabel(x_label, fontweight=fontweight)
            ax.set_ylabel(y_label, fontweight=fontweight)
            
            dl_display = _standardize_model_name(dl_model)
            ml_display = _standardize_model_name(ml_model)
            ax.set_title(f'{dl_display} + {ml_display}', fontsize=12, pad=8, fontweight=fontweight)
            ax.grid(True, linestyle='--', alpha=0.3, zorder=0)
            
            # Set axis limits if provided
            if x_lim is not None:
                ax.set_xlim(x_lim)
            if y_lim is not None:
                ax.set_ylim(y_lim)
            
            # Set axis tick spacing if provided
            if x_locator is not None:
                if isinstance(x_locator, int):
                    ax.xaxis.set_major_locator(MaxNLocator(nbins=x_locator))
                else:
                    ax.xaxis.set_major_locator(MultipleLocator(x_locator))
            
            if y_locator is not None:
                if isinstance(y_locator, int):
                    ax.yaxis.set_major_locator(MaxNLocator(nbins=y_locator))
                else:
                    ax.yaxis.set_major_locator(MultipleLocator(y_locator))
            
            # Legend only for top-left subplot
            if row_idx == 0 and col_idx == 0:
                ax.legend(loc='lower right', framealpha=0.9, fontsize=9)
        
        axes.append(row_axes)
    
    # Add colorbar if color_metric is specified
    if color_metric:
        sm = ScalarMappable(cmap=colormap, norm=norm)
        sm.set_array([])
        
        cbar_ax = fig.add_axes([0.93, 0.15, 0.02, 0.7])
        cbar = fig.colorbar(sm, cax=cbar_ax)
        color_label = _standardize_metric_name(color_metric)
        fontweight_cbar = 'bold' if bold else 'normal'
        cbar.set_label(color_label, fontsize=12, rotation=270, labelpad=20, fontweight=fontweight_cbar)
        cbar.ax.tick_params(labelsize=10)
        cbar.ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Figure saved to: {save_path}")
    
    return fig, axes
