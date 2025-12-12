"""
Shared plotting functions for Figure 2 and Figure 4

Contains the core plotting functions:
- plot_capacity_strategy_comparison: Training curves (epoch vs R²)
- plot_represent_parameter_vs_test_error: Parameter space (params vs R²)

Ported from OG_transformer.appendix.capacity_analysis and OG_transformer.appendix.grid_performance
"""

import os
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
from matplotlib.ticker import MaxNLocator, FuncFormatter
from scipy.interpolate import interp1d, PchipInterpolator
from scipy.signal import savgol_filter
from typing import Tuple, Optional, List, Dict


# =============================================================================
# Strategy mapping and colors (shared constants)
# =============================================================================

STRATEGY_MAPPING = {
    'train': 'Training',
    'Sample': 'Time-wise',
    'Site': 'Site-wise',
    'Grid': 'Region-wise',
    'Time': 'Point-wise',
    'OrderTime': 'Sequential',
    'Spatiotemporal': 'Spatiotemporal'
}

STRATEGY_COLORS = {
    'train': 'grey',
    'Sample': '#2E86AB',
    'Site': '#A23B72',
    'Grid': '#2ca02c',
    'Time': '#d62728',
    'OrderTime': '#8c564b'
}


# =============================================================================
# Helper: Curve fitting
# =============================================================================

def _apply_fit_capacity(x_vals, y_vals, std_vals, fit_type, degree):
    """Apply curve fitting for capacity analysis (epoch data)."""
    x_fit = x_vals
    
    # Extend range slightly
    x_range = x_fit.max() - x_fit.min()
    extend_ratio = 0.02
    x_min_ext = x_fit.min() - x_range * extend_ratio
    x_max_ext = x_fit.max() + x_range * extend_ratio
    
    if fit_type == 'peak_decay':
        linear_interp = interp1d(x_fit, y_vals, kind='linear', fill_value='extrapolate')
        linear_interp_upper = interp1d(x_fit, y_vals + std_vals, kind='linear', fill_value='extrapolate')
        linear_interp_lower = interp1d(x_fit, y_vals - std_vals, kind='linear', fill_value='extrapolate')
        
        x_smooth = np.linspace(x_min_ext, x_max_ext, 100)
        y_line = linear_interp(x_smooth)
        y_upper = linear_interp_upper(x_smooth)
        y_lower = linear_interp_lower(x_smooth)
        
        window = min(31, len(x_smooth) if len(x_smooth) % 2 == 1 else len(x_smooth) - 1)
        polyorder = min(3, window - 1)
        y_smooth = savgol_filter(y_line, window_length=window, polyorder=polyorder)
        y_upper = savgol_filter(y_upper, window_length=window, polyorder=polyorder)
        y_lower = savgol_filter(y_lower, window_length=window, polyorder=polyorder)
        
    elif fit_type == 'monotonic':
        pchip = PchipInterpolator(x_fit, y_vals)
        pchip_upper = PchipInterpolator(x_fit, y_vals + std_vals)
        pchip_lower = PchipInterpolator(x_fit, y_vals - std_vals)
        
        x_smooth = np.linspace(x_min_ext, x_max_ext, 100)
        y_smooth = pchip(x_smooth)
        y_upper = pchip_upper(x_smooth)
        y_lower = pchip_lower(x_smooth)
        
    else:  # 'polynomial' or default
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
            interp_func = interp1d(x_fit, y_vals, kind='linear', fill_value='extrapolate')
            x_smooth = np.linspace(x_min_ext, x_max_ext, 100)
            y_smooth = interp_func(x_smooth)
            y_upper = y_smooth + std_vals.mean()
            y_lower = y_smooth - std_vals.mean()
    
    return x_smooth, y_smooth, y_upper, y_lower


def _apply_fit_parameter(x_vals, y_vals, std_vals, use_log_scale, fit_type, degree, x_full_range):
    """Apply curve fitting for parameter analysis."""
    if use_log_scale:
        x_fit = np.log10(x_vals)
        x_full_fit = np.log10(x_full_range)
    else:
        x_fit = x_vals
        x_full_fit = x_full_range
    
    x_range = x_full_fit.max() - x_full_fit.min()
    extend_ratio = 0.02
    x_min_ext = x_full_fit.min() - x_range * extend_ratio
    x_max_ext = x_full_fit.max() + x_range * extend_ratio
    
    if fit_type == 'peak_decay':
        linear_interp = interp1d(x_fit, y_vals, kind='linear', fill_value='extrapolate')
        linear_interp_upper = interp1d(x_fit, y_vals + std_vals, kind='linear', fill_value='extrapolate')
        linear_interp_lower = interp1d(x_fit, y_vals - std_vals, kind='linear', fill_value='extrapolate')
        
        x_smooth = np.linspace(x_min_ext, x_max_ext, 52)
        y_line = linear_interp(x_smooth)
        y_upper = linear_interp_upper(x_smooth)
        y_lower = linear_interp_lower(x_smooth)
        
        window = min(31, len(x_smooth) if len(x_smooth) % 2 == 1 else len(x_smooth) - 1)
        polyorder = min(3, window - 1)
        y_smooth = savgol_filter(y_line, window_length=window, polyorder=polyorder)
        y_upper = savgol_filter(y_upper, window_length=window, polyorder=polyorder)
        y_lower = savgol_filter(y_lower, window_length=window, polyorder=polyorder)
        
    elif fit_type == 'monotonic':
        pchip = PchipInterpolator(x_fit, y_vals)
        pchip_upper = PchipInterpolator(x_fit, y_vals + std_vals)
        pchip_lower = PchipInterpolator(x_fit, y_vals - std_vals)
        
        x_smooth = np.linspace(x_min_ext, x_max_ext, 52)
        y_smooth = pchip(x_smooth)
        y_upper = pchip_upper(x_smooth)
        y_lower = pchip_lower(x_smooth)
        
    else:  # 'polynomial'
        try:
            poly_coeffs = np.polyfit(x_fit, y_vals, degree)
            poly_func = np.poly1d(poly_coeffs)
            
            x_smooth = np.linspace(x_min_ext, x_max_ext, 52)
            y_smooth = poly_func(x_smooth)
            
            poly_upper = np.polyfit(x_fit, y_vals + std_vals, degree)
            poly_lower = np.polyfit(x_fit, y_vals - std_vals, degree)
            y_upper = np.poly1d(poly_upper)(x_smooth)
            y_lower = np.poly1d(poly_lower)(x_smooth)
        except:
            interp_func = interp1d(x_fit, y_vals, kind='linear', fill_value='extrapolate')
            x_smooth = np.linspace(x_min_ext, x_max_ext, 52)
            y_smooth = interp_func(x_smooth)
            y_upper = y_smooth + std_vals.mean()
            y_lower = y_smooth - std_vals.mean()
    
    if use_log_scale:
        x_plot = 10**x_smooth
    else:
        x_plot = x_smooth
    
    return x_plot, y_smooth, y_upper, y_lower


# =============================================================================
# plot_capacity_strategy_comparison
# =============================================================================

def plot_capacity_strategy_comparison(model_list, project_path,
                                       left_metrics=None,
                                       right_metrics=None,
                                       assign_train=None,
                                       degree=2,
                                       exclude_first_n_points=0,
                                       confidence_band=True,
                                       show_points=True,
                                       ratio=0.5,
                                       scale=1.0,
                                       left_y_lim=None,
                                       right_y_lim=None,
                                       max_locator=None,
                                       fit_types=None,
                                       save_svg=None,
                                       ax=None,
                                       show_legend=True,
                                       show_title=True,
                                       show_xlabel=True,
                                       show_ylabel=True,
                                       reduce_points=False):
    """
    Plot training capacity curves for multiple strategies with dual y-axis.
    
    Parameters
    ----------
    model_list : list
        List of model names to plot
    project_path : str
        Directory containing capacity CSV files
    left_metrics : list of str
        Strategies to plot on left y-axis, e.g., ["Sample"]
    right_metrics : list of str
        Strategies to plot on right y-axis, e.g., ["Site", "Grid"]
    assign_train : str, optional
        Strategy name to use for extracting train_r2
    degree : int
        Polynomial degree for curve fitting
    exclude_first_n_points : int
        Number of initial data points to exclude when fitting
    confidence_band : bool
        Whether to show confidence bands
    show_points : bool
        Whether to show original data points as markers
    ratio, scale : float
        Layout parameters
    left_y_lim, right_y_lim : tuple or list
        Y-axis limits
    max_locator : tuple
        Maximum number of y-axis ticks (left, right)
    fit_types : dict
        Fitting type for specific strategies
    ax : matplotlib.axes.Axes
        External axes to plot on (single model only)
    show_legend, show_title, show_xlabel, show_ylabel : bool
        Display options
    reduce_points : bool
        If True, subsample data points
    
    Returns
    -------
    fig, axes or (None, ax)
    """
    if left_metrics is None or right_metrics is None:
        raise ValueError("Both left_metrics and right_metrics must be provided")
    
    def get_ylim_for_model(ylim_param, model_idx):
        if ylim_param is None:
            return None
        elif isinstance(ylim_param, tuple):
            return ylim_param
        elif isinstance(ylim_param, list):
            return ylim_param[model_idx] if model_idx < len(ylim_param) else None
        return None
    
    def plot_strategy_on_axis(axis, strategy, agg_df, default_fit_type='-', marker='o'):
        if agg_df is None or len(agg_df) == 0:
            return
        
        agg_df.columns = ['epoch', 'mean', 'std']
        agg_df['std'] = agg_df['std'].fillna(0)
        
        color = STRATEGY_COLORS.get(strategy, 'black')
        display_name = STRATEGY_MAPPING.get(strategy, strategy)
        fit_type_raw = fit_types.get(strategy, default_fit_type) if fit_types else default_fit_type
        
        # Parse fit_type
        if fit_type_raw.endswith('--'):
            fit_type = fit_type_raw[:-2]
            linestyle = '--'
        elif fit_type_raw.endswith('-') and fit_type_raw != '-':
            fit_type = fit_type_raw[:-1]
            linestyle = '-'
        else:
            fit_type = fit_type_raw
            linestyle = '-'
        
        x_vals = agg_df['epoch'].values
        y_vals = agg_df['mean'].values
        std_vals = agg_df['std'].values
        
        if fit_type in ['-', '--']:
            axis.plot(x_vals, y_vals, color=color, linewidth=2, alpha=0.8, linestyle=fit_type, label=display_name)
            if confidence_band:
                axis.fill_between(x_vals, y_vals - std_vals, y_vals + std_vals, color=color, alpha=0.2)
        else:
            # Exclude first n points for fitting
            if exclude_first_n_points > 0 and len(x_vals) > exclude_first_n_points:
                x_fit = x_vals[exclude_first_n_points:]
                y_fit = y_vals[exclude_first_n_points:]
                std_fit = std_vals[exclude_first_n_points:]
            else:
                x_fit, y_fit, std_fit = x_vals, y_vals, std_vals
            
            x_plot, y_smooth, y_upper, y_lower = _apply_fit_capacity(x_fit, y_fit, std_fit, fit_type, degree)
            
            axis.plot(x_plot, y_smooth, linestyle, color=color, linewidth=2, alpha=0.8, label=display_name)
            if confidence_band:
                axis.fill_between(x_plot, y_lower, y_upper, color=color, alpha=0.2)
        
        if show_points:
            axis.plot(x_vals, y_vals, marker, color=color, markersize=4, alpha=0.6)
    
    n_models = len(model_list)
    use_external_ax = (ax is not None)
    
    if use_external_ax:
        if n_models != 1:
            raise ValueError(f"When using external ax, model_list must contain exactly 1 model. Got {n_models}.")
        model_name = model_list[0]
        fig = None
        axes_to_iterate = [(ax, 0)]
    else:
        n_cols = 3
        n_rows = max(1, (n_models + n_cols - 1) // n_cols)
        
        width_per_subplot = 4.4
        height_per_subplot = width_per_subplot * ratio
        total_width = width_per_subplot * n_cols * scale
        total_height = height_per_subplot * n_rows * scale
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(total_width, total_height))
        
        if n_rows == 1 and n_cols == 1:
            axes = np.array([[axes]])
        elif n_rows == 1:
            axes = axes.reshape(1, -1)
        elif n_cols == 1:
            axes = axes.reshape(-1, 1)
        
        axes_to_iterate = []
        plot_idx = 0
        for i in range(n_rows):
            for j in range(n_cols):
                if plot_idx >= n_models:
                    axes[i, j].axis('off')
                    continue
                axes_to_iterate.append((axes[i, j], plot_idx))
                plot_idx += 1
    
    for ax_current, plot_idx in axes_to_iterate:
        model_name = model_list[plot_idx] if not use_external_ax else model_list[0]
        current_left_ylim = get_ylim_for_model(left_y_lim, plot_idx)
        current_right_ylim = get_ylim_for_model(right_y_lim, plot_idx)
        
        csv_path = os.path.join(project_path, f"{model_name}.csv")
        
        if not os.path.exists(csv_path):
            ax_current.text(0.5, 0.5, f"{model_name}\nNo data", ha='center', va='center', transform=ax_current.transAxes)
            if show_title:
                ax_current.set_title(model_name)
            continue
        
        df = pd.read_csv(csv_path)
        
        if len(df) == 0:
            ax_current.text(0.5, 0.5, f"{model_name}\nNo data", ha='center', va='center', transform=ax_current.transAxes)
            continue
        
        def get_strategy_data(strategy):
            if strategy == 'train':
                if assign_train:
                    strategy_df = df[df['strategy'] == assign_train]
                    if len(strategy_df) == 0:
                        return None
                    result_df = strategy_df.groupby('epoch')['train_r2'].agg(['mean', 'std']).reset_index()
                else:
                    strategy_df = df[df['strategy'] == 'train']
                    if len(strategy_df) == 0:
                        return None
                    result_df = strategy_df.groupby('epoch')['val_r2'].agg(['mean', 'std']).reset_index()
            else:
                strategy_df = df[df['strategy'] == strategy]
                if len(strategy_df) == 0:
                    return None
                result_df = strategy_df.groupby('epoch')['val_r2'].agg(['mean', 'std']).reset_index()
            
            if reduce_points and len(result_df) > 1:
                indices = list(range(0, len(result_df), 2))
                result_df = result_df.iloc[indices].reset_index(drop=True)
            
            return result_df
        
        ax2_current = ax_current.twinx()
        
        for strategy in left_metrics:
            plot_strategy_on_axis(ax_current, strategy, get_strategy_data(strategy), default_fit_type='-', marker='o')
        
        for strategy in right_metrics:
            plot_strategy_on_axis(ax2_current, strategy, get_strategy_data(strategy), default_fit_type='--', marker='s')
        
        if show_xlabel:
            ax_current.set_xlabel('Epoch', fontsize=11)
        
        if show_ylabel:
            left_names = [STRATEGY_MAPPING.get(s, s) for s in left_metrics]
            ax_current.set_ylabel(' + '.join(left_names), fontsize=11, color='black')
            right_names = [STRATEGY_MAPPING.get(s, s) for s in right_metrics]
            ax2_current.set_ylabel(' + '.join(right_names), fontsize=11, color='black')
        
        if show_title:
            ax_current.set_title(f'{model_name}', fontsize=12, fontweight='bold')
        
        if current_left_ylim is not None:
            ax_current.set_ylim(current_left_ylim)
        if current_right_ylim is not None:
            ax2_current.set_ylim(current_right_ylim)
        
        if max_locator is not None:
            left_n, right_n = max_locator
            ax_current.yaxis.set_major_locator(MaxNLocator(nbins=left_n, prune=None))
            ax_current.yaxis.set_major_formatter(FuncFormatter(lambda x, p: f'{x:.2f}'))
            ax2_current.yaxis.set_major_locator(MaxNLocator(nbins=right_n, prune=None))
            ax2_current.yaxis.set_major_formatter(FuncFormatter(lambda x, p: f'{x:.2f}'))
        
        ax_current.margins(x=0)
        ax_current.grid(True, alpha=0.3)
        
        if show_legend:
            lines1, labels1 = ax_current.get_legend_handles_labels()
            lines2, labels2 = ax2_current.get_legend_handles_labels()
            ax_current.legend(lines1 + lines2, labels1 + labels2, loc='best', fontsize=9, framealpha=0.9)
    
    if not use_external_ax:
        plt.tight_layout()
        if save_svg:
            plt.savefig(save_svg, format='svg', bbox_inches='tight')
            print(f"✓ Saved SVG plot to: {save_svg}")
        return fig, axes
    else:
        return None, ax


# =============================================================================
# plot_represent_parameter_vs_test_error
# =============================================================================

def plot_represent_parameter_vs_test_error(model_list, project_path,
                                           x_axis='parameter_count',
                                           metric_pairs=None,
                                           left_metrics=None,
                                           right_metrics=None,
                                           sample_size='full',
                                           degree=2,
                                           uncertainty='band',
                                           confidence_band=True,
                                           show_points=True,
                                           ratio=0.68,
                                           scale=1.0,
                                           left_y_lim=None,
                                           right_y_lim=None,
                                           max_locator=None,
                                           fit_types=None,
                                           ax=None,
                                           show_legend=True,
                                           show_title=True,
                                           show_xlabel=True,
                                           show_ylabel=True,
                                           use_folds=None,
                                           use_log_scale=None,
                                           exclude_point_idx=None,
                                           metric="r2_score"):
    """
    Plot parameter vs test error for multiple splitting strategies.
    
    Parameters
    ----------
    model_list : list
        List of model names to plot
    project_path : str
        Directory containing parameter CSV files
    x_axis : str
        'parameter_count' (log scale) or 'original_param' (linear)
    left_metrics, right_metrics : list of str
        Strategies for left and right y-axes
    sample_size : int or 'full'
        Which sample size to plot
    degree : int
        Polynomial degree for curve fitting
    uncertainty : str
        'band' or 'point'
    confidence_band, show_points : bool
        Display options
    left_y_lim, right_y_lim : tuple or list
        Y-axis limits
    max_locator : tuple
        Maximum ticks (left, right)
    fit_types : dict
        Fitting type per strategy
    ax : Axes
        External axes (single model only)
    use_folds : list
        Fold indices to use
    exclude_point_idx : list
        Indices to exclude from fitting
    metric : str
        Column name for metric
    
    Returns
    -------
    fig, axes or (None, ax)
    """
    if left_metrics is None or right_metrics is None:
        raise ValueError("Both left_metrics and right_metrics must be provided")
    
    # Determine log scale
    if use_log_scale is None:
        use_log_scale = (x_axis == 'parameter_count')
    
    def get_ylim_for_model(ylim_param, model_idx):
        if ylim_param is None:
            return None
        elif isinstance(ylim_param, tuple):
            return ylim_param
        elif isinstance(ylim_param, list):
            return ylim_param[model_idx] if model_idx < len(ylim_param) else None
        return None
    
    def plot_strategy_on_axis(axis, strategy, strategy_df, df_full, marker='o'):
        if len(strategy_df) == 0:
            return
        
        stats = strategy_df.groupby('x_value').agg({metric: ['mean', 'std']}).reset_index()
        stats.columns = ['x_value', 'mean', 'std']
        
        stats_for_fit = stats.copy()
        if exclude_point_idx is not None:
            indices_to_drop = [i if i >= 0 else len(stats_for_fit) + i for i in exclude_point_idx]
            stats_for_fit = stats_for_fit.drop(stats_for_fit.index[indices_to_drop], errors='ignore')
        
        color = STRATEGY_COLORS.get(strategy, 'black')
        label = STRATEGY_MAPPING.get(strategy, strategy)
        fit_type = fit_types.get(strategy, 'polynomial') if fit_types else 'polynomial'
        
        if uncertainty == 'point':
            axis.plot(stats['x_value'], stats['mean'], marker, color=color, label=label, markersize=6)
        else:
            x_vals = stats_for_fit['x_value'].values
            y_vals = stats_for_fit['mean'].values
            std_vals = stats_for_fit['std'].values
            x_full = stats['x_value'].values
            
            if fit_type in ['-', '--']:
                axis.plot(x_vals, y_vals, fit_type, color=color, label=label, linewidth=2)
                if confidence_band:
                    axis.fill_between(x_vals, y_vals - std_vals, y_vals + std_vals, alpha=0.2, color=color)
            else:
                x_plot, y_smooth, y_upper, y_lower = _apply_fit_parameter(x_vals, y_vals, std_vals, use_log_scale, fit_type, degree, x_full)
                
                axis.plot(x_plot, y_smooth, '-', color=color, label=label, linewidth=2)
                if confidence_band:
                    axis.fill_between(x_plot, y_lower, y_upper, alpha=0.2, color=color)
                
                if show_points:
                    axis.plot(stats['x_value'].values, stats['mean'].values, marker, color=color, markersize=4, alpha=0.6)
    
    n_models = len(model_list)
    use_external_ax = (ax is not None)
    
    if use_external_ax:
        if n_models != 1:
            raise ValueError(f"When using external ax, model_list must have 1 model. Got {n_models}.")
        fig = None
        axes_to_iterate = [(ax, 0)]
    else:
        n_cols = min(3, n_models)
        n_rows = max(1, (n_models + n_cols - 1) // n_cols)
        
        width_per_subplot = 4.4
        height_per_subplot = width_per_subplot * ratio
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(width_per_subplot * n_cols * scale, height_per_subplot * n_rows * scale))
        
        if n_rows == 1 and n_cols == 1:
            axes = np.array([[axes]])
        elif n_rows == 1:
            axes = axes.reshape(1, -1)
        elif n_cols == 1:
            axes = axes.reshape(-1, 1)
        
        axes_to_iterate = []
        plot_idx = 0
        for i in range(n_rows):
            for j in range(n_cols):
                if plot_idx >= n_models:
                    axes[i, j].axis('off')
                    continue
                axes_to_iterate.append((axes[i, j], plot_idx))
                plot_idx += 1
    
    for ax_current, plot_idx in axes_to_iterate:
        model_name = model_list[plot_idx] if not use_external_ax else model_list[0]
        current_left_ylim = get_ylim_for_model(left_y_lim, plot_idx)
        current_right_ylim = get_ylim_for_model(right_y_lim, plot_idx)
        
        csv_path = os.path.join(project_path, f"{model_name}.csv")
        
        if not os.path.exists(csv_path):
            ax_current.text(0.5, 0.5, f"{model_name}\nNo data", ha='center', va='center', transform=ax_current.transAxes)
            if show_title:
                ax_current.set_title(model_name)
            continue
        
        df = pd.read_csv(csv_path)
        
        if len(df) == 0:
            ax_current.text(0.5, 0.5, f"{model_name}\nNo data", ha='center', va='center', transform=ax_current.transAxes)
            continue
        
        # Filter by sample_size
        if 'sample_size' in df.columns and sample_size != 'full':
            df = df[df['sample_size'] == sample_size]
        
        # Filter by folds
        if use_folds is not None and 'fold' in df.columns:
            df = df[df['fold'].isin(use_folds)]
        
        # Create x_value from parameter_count or original param
        if 'parameter_number' in df.columns:
            df['x_value'] = df['parameter_number']
        elif 'parameter_count' in df.columns:
            df['x_value'] = df['parameter_count']
        else:
            ax_current.text(0.5, 0.5, f"{model_name}\nNo parameter column", ha='center', va='center', transform=ax_current.transAxes)
            continue
        
        ax2_current = ax_current.twinx()
        
        # Plot left metrics
        for strategy in left_metrics:
            if 'strategy' in df.columns:
                strategy_df = df[df['strategy'] == strategy].copy()
            else:
                strategy_df = df.copy()
            plot_strategy_on_axis(ax_current, strategy, strategy_df, df, marker='o')
        
        # Plot right metrics
        for strategy in right_metrics:
            if 'strategy' in df.columns:
                strategy_df = df[df['strategy'] == strategy].copy()
            else:
                strategy_df = df.copy()
            plot_strategy_on_axis(ax2_current, strategy, strategy_df, df, marker='s')
        
        if show_xlabel:
            ax_current.set_xlabel('Parameter Count', fontsize=11)
        
        if show_ylabel:
            left_names = [STRATEGY_MAPPING.get(s, s) for s in left_metrics]
            ax_current.set_ylabel(' + '.join(left_names), fontsize=11, color='black')
            right_names = [STRATEGY_MAPPING.get(s, s) for s in right_metrics]
            ax2_current.set_ylabel(' + '.join(right_names), fontsize=11, color='black')
        
        if show_title:
            ax_current.set_title(f'{model_name}', fontsize=12, fontweight='bold')
        
        if current_left_ylim is not None:
            ax_current.set_ylim(current_left_ylim)
        if current_right_ylim is not None:
            ax2_current.set_ylim(current_right_ylim)
        
        if use_log_scale:
            ax_current.set_xscale('log')
        
        if max_locator is not None:
            left_n, right_n = max_locator
            ax_current.yaxis.set_major_locator(MaxNLocator(nbins=left_n, prune=None))
            ax_current.yaxis.set_major_formatter(FuncFormatter(lambda x, p: f'{x:.2f}'))
            ax2_current.yaxis.set_major_locator(MaxNLocator(nbins=right_n, prune=None))
            ax2_current.yaxis.set_major_formatter(FuncFormatter(lambda x, p: f'{x:.2f}'))
        
        ax_current.margins(x=0)
        ax_current.grid(True, alpha=0.3)
        
        if show_legend:
            lines1, labels1 = ax_current.get_legend_handles_labels()
            lines2, labels2 = ax2_current.get_legend_handles_labels()
            ax_current.legend(lines1 + lines2, labels1 + labels2, loc='best', fontsize=9, framealpha=0.9)
    
    if not use_external_ax:
        plt.tight_layout()
        return fig, axes
    else:
        return None, ax
