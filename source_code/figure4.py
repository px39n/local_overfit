"""
Figure 4: OG vs Standard Model Comparison

This module provides three comparison functions:

1. figure4_capacity: Capacity Analysis (epoch training curves) - 2x2 layout
   - Top-left: ResNet (standard, epoch vs R²)
   - Bottom-left: Transformer (standard, epoch vs R²)
   - Top-right: OG-ResNet (OG version, epoch vs R²)
   - Bottom-right: OG-Transformer (OG version, epoch vs R²)

2. figure4_capacity_13: Capacity Analysis for OG models only - 1x3 layout
   - Left: OG-MLP (epoch vs R²)
   - Middle: OG-ResNet (epoch vs R²)
   - Right: OG-Transformer (epoch vs R²)

3. figure4_parameter13: Parameter Space Analysis (parameter count vs R²) - 1x3 layout

Ported from OG_transformer.plot.figure4, using shared functions from appendix module.
"""

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from sklearn.metrics import r2_score

# Import shared plotting functions from appendix
from .appendix import (
    plot_capacity_strategy_comparison,
    plot_represent_parameter_vs_test_error
)


def figure4_capacity(epoch_project_path="data/appendix/5-Fold-Epoch",
                     figsize=(12, 10),
                     config=1,
                     save_path=None):
    """
    Create a 2x2 comparison figure: Standard models vs OG models (capacity).
    
    Parameters
    ----------
    epoch_project_path : str
        Path to epoch training curves results
    figsize : tuple
        Figure size (width, height)
    config : int
        Configuration mode: 1 (no confidence band) or 2 (with confidence band)
    save_path : str, optional
        Path to save the figure. If None, figure is not saved.
    
    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure object
    axes : dict
        Dictionary of axes {'resnet': ax1, 'transformer': ax2, 'og_resnet': ax3, 'og_transformer': ax4}
    """
    # Create figure and GridSpec
    fig = plt.figure(figsize=figsize)
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.6)
    
    # Create axes
    ax_resnet = fig.add_subplot(gs[0, 0])           # Top-left: Standard ResNet
    ax_transformer = fig.add_subplot(gs[1, 0])      # Bottom-left: Standard Transformer
    ax_og_resnet = fig.add_subplot(gs[0, 1])        # Top-right: OG-ResNet
    ax_og_transformer = fig.add_subplot(gs[1, 1])   # Bottom-right: OG-Transformer
    
    confidence_band = (config == 2)
    
    # Plot Standard ResNet (top-left)
    plot_capacity_strategy_comparison(
        ['resnet'],
        epoch_project_path,
        assign_train="Sample",
        left_metrics=["Sample"],
        right_metrics=["Site", "Grid"],
        left_y_lim=(0.46, 0.65),
        right_y_lim=(0.46, 0.65),
        fit_types={
            'Sample': 'polynomial',
            'Site': 'peak_decay',
            'Grid': 'peak_decay'
        },
        confidence_band=confidence_band,
        exclude_first_n_points=1,
        ratio=0.8,
        scale=0.9,
        max_locator=(6, 6),
        ax=ax_resnet,
        show_legend=False,
        show_xlabel=False,
        show_title=False,
        show_ylabel=True
    )
    
    # Plot Standard Transformer (bottom-left)
    plot_capacity_strategy_comparison(
        ['transformer'],
        epoch_project_path,
        assign_train="Sample",
        left_metrics=["Sample"],
        degree=2,
        right_metrics=["Site", "Grid"],
        left_y_lim=(0.5, 0.65),
        right_y_lim=(0.5, 0.65),
        confidence_band=confidence_band,
        fit_types={
            'Sample': 'polynomial',
            'Site': 'peak_decay',
            'Grid': 'peak_decay'
        },
        exclude_first_n_points=1,
        show_points=True,
        ratio=0.7,
        scale=0.9,
        max_locator=(6, 6),
        ax=ax_transformer,
        show_legend=False,
        show_xlabel=False,
        show_title=False,
        show_ylabel=True
    )
    
    # Plot OG-ResNet (top-right)
    plot_capacity_strategy_comparison(
        ['og_resnet'],
        epoch_project_path,
        assign_train="Sample",
        left_metrics=["Sample"],
        right_metrics=["Site", "Grid"],
        left_y_lim=(0.46, 0.65),
        right_y_lim=(0.46, 0.65),
        fit_types={
            'Sample': 'polynomial',
            'Site': 'peak_decay',
            'Grid': 'peak_decay'
        },
        confidence_band=confidence_band,
        exclude_first_n_points=1,
        ratio=0.8,
        scale=0.9,
        max_locator=(6, 6),
        ax=ax_og_resnet,
        show_legend=False,
        show_xlabel=False,
        show_title=False,
        show_ylabel=True
    )
    
    # Plot OG-Transformer (bottom-right)
    plot_capacity_strategy_comparison(
        ['og_transformer'],
        epoch_project_path,
        assign_train="Sample",
        left_metrics=["Sample"],
        degree=2,
        right_metrics=["Site", "Grid"],
        left_y_lim=(0.5, 0.65),
        right_y_lim=(0.5, 0.65),
        confidence_band=confidence_band,
        fit_types={
            'Sample': 'polynomial',
            'Site': 'peak_decay',
            'Grid': 'peak_decay'
        },
        exclude_first_n_points=1,
        show_points=True,
        ratio=0.7,
        scale=0.9,
        max_locator=(6, 6),
        ax=ax_og_transformer,
        show_legend=False,
        show_xlabel=False,
        show_title=False,
        show_ylabel=True
    )
    
    # Adjust layout
    plt.tight_layout()
    
    # Save if path provided
    if save_path is not None:
        if save_path.lower().endswith('.svg'):
            fig.savefig(save_path, format='svg', bbox_inches='tight')
        else:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Figure 4 (Capacity) saved to: {save_path}")
    
    # Return figure and axes dictionary
    axes_dict = {
        'resnet': ax_resnet,
        'transformer': ax_transformer,
        'og_resnet': ax_og_resnet,
        'og_transformer': ax_og_transformer
    }
    
    return fig, axes_dict


def figure4_parameter13(param_project_path="data/appendix/5CV_validation_vs_parameter",
                        models=['og_mlp', 'og_resnet', 'og_transformer'],
                        figsize=(15, 5),
                        sample_size='full',
                        save_path=None,
                        use_folds=None,
                        exclude_point_idx=None,
                        y_lim_1=None,
                        y_lim_2=None,
                        y_lim_3=None,
                        xlabel=False,
                        title=False,
                        show_legend=False,
                        max_locator=(6, 5),
                        metric="r2_score"):
    """
    Create a 1x3 comparison figure for parameter space.

    Parameters
    ----------
    param_project_path : str
        Path to parameter space analysis results
    models : list of str, default=['og_mlp', 'og_resnet', 'og_transformer']
        List of 3 model names to plot (left, middle, right).
        Examples: ['mlp', 'resnet', 'transformer'] or ['og_mlp', 'og_resnet', 'og_transformer']
    figsize : tuple
        Figure size (width, height)
    sample_size : str or int
        Sample size to analyze ('full' or specific number)
    save_path : str, optional
        Path to save the figure. If None, figure is not saved.
    use_folds : list of int, optional
        Which folds to include when aggregating
    exclude_point_idx : list of int, optional
        Indices to exclude from fitting on x-axis (still shown)
    y_lim_1/2/3 : tuple of (left_lim, right_lim), optional
        Per-subplot y limits; each lim is (min, max)
    """
    if len(models) != 3:
        raise ValueError(f"models must contain exactly 3 model names, got {len(models)}")
    
    # Create figure and GridSpec
    fig = plt.figure(figsize=figsize)
    gs = gridspec.GridSpec(1, 3, figure=fig, hspace=0.3, wspace=0.6)

    # Axes: left, middle, right
    ax_left = fig.add_subplot(gs[0, 0])
    ax_middle = fig.add_subplot(gs[0, 1])
    ax_right = fig.add_subplot(gs[0, 2])

    # Defaults for y limits if not provided
    if y_lim_1 is not None:
        left_lim_1, right_lim_1 = y_lim_1
    else:
        left_lim_1, right_lim_1 = (0.50, 0.65), (0.50, 0.65)

    if y_lim_2 is not None:
        left_lim_2, right_lim_2 = y_lim_2
    else:
        left_lim_2, right_lim_2 = (0.46, 0.65), (0.46, 0.65)

    if y_lim_3 is not None:
        left_lim_3, right_lim_3 = y_lim_3
    else:
        left_lim_3, right_lim_3 = (0.50, 0.65), (0.50, 0.65)

    # Left subplot (models[0])
    plot_represent_parameter_vs_test_error(
        [models[0]],
        param_project_path,
        x_axis='parameter_count',
        sample_size=sample_size,
        left_metrics=["Sample"],
        right_metrics=["Site", "Grid"],
        degree=2,
        uncertainty='band',
        use_folds=use_folds,
        exclude_point_idx=exclude_point_idx,
        left_y_lim=[left_lim_1],
        right_y_lim=[right_lim_1],
        ratio=0.7,
        fit_types={
            'Sample': 'polynomial',
            'Site': 'polynomial',
            'Grid': 'polynomial'
        },
        max_locator=max_locator,
        ax=ax_left,
        show_legend=show_legend,
        show_title=title,
        show_xlabel=xlabel,
        show_ylabel=True,
        metric=metric
    )

    # Middle subplot (models[1])
    plot_represent_parameter_vs_test_error(
        [models[1]],
        param_project_path,
        x_axis='parameter_count',
        sample_size=sample_size,
        left_metrics=["Sample"],
        right_metrics=["Site", "Grid"],
        degree=2,
        uncertainty='band',
        use_folds=use_folds,
        exclude_point_idx=exclude_point_idx,
        left_y_lim=[left_lim_2],
        right_y_lim=[right_lim_2],
        ratio=0.8,
        fit_types={
            'Sample': 'polynomial',
            'Site': 'polynomial',
            'Grid': 'polynomial'
        },
        max_locator=max_locator,
        ax=ax_middle,
        show_legend=False,
        show_title=title,
        show_xlabel=xlabel,
        show_ylabel=True,
        metric=metric
    )

    # Right subplot (models[2])
    plot_represent_parameter_vs_test_error(
        [models[2]],
        param_project_path,
        x_axis='parameter_count',
        sample_size=sample_size,
        left_metrics=["Sample"],
        right_metrics=["Site", "Grid"],
        degree=2,
        uncertainty='band',
        use_folds=use_folds,
        exclude_point_idx=exclude_point_idx,
        left_y_lim=[left_lim_3],
        right_y_lim=[right_lim_3],
        ratio=0.7,
        fit_types={
            'Sample': 'polynomial',
            'Site': 'polynomial',
            'Grid': 'polynomial'
        },
        max_locator=max_locator,
        ax=ax_right,
        show_legend=False,
        show_title=title,
        show_xlabel=xlabel,
        show_ylabel=True,
        metric=metric
    )

    # Layout
    plt.tight_layout()

    # Save if requested
    if save_path is not None:
        if save_path.lower().endswith('.svg'):
            fig.savefig(save_path, format='svg', bbox_inches='tight')
        else:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Figure 4 (Parameter 1x3) saved to: {save_path}")

    axes = {
        models[0]: ax_left,
        models[1]: ax_middle,
        models[2]: ax_right,
    }
    return fig, axes


def figure4_capacity_13(epoch_project_path="data/appendix/5-Fold-Epoch",
                        figsize=(15, 5),
                        config=1,
                        save_path=None,
                        title=False,
                        xlabel=False,
                        show_legend=False,
                        og=True,
                        left_y_lim=(0.5, 0.65),
                        right_y_lim=(0.5, 0.65),
                        fit_types=None,
                        reduce_points=False):
    """
    Create a 1x3 comparison figure: OG-MLP, OG-ResNet, OG-Transformer (capacity).
    
    Parameters
    ----------
    epoch_project_path : str
        Path to epoch training curves results
    figsize : tuple
        Figure size (width, height)
    config : int
        Configuration mode: 1 (no confidence band) or 2 (with confidence band)
    save_path : str, optional
        Path to save the figure. If None, figure is not saved.
    title : bool, default=False
        Whether to show subplot titles
    xlabel : bool, default=False
        Whether to show x-axis labels
    show_legend : bool, default=False
        Whether to show legend
    og : bool, default=True
        If True, plot OG models; if False, plot standard models
    left_y_lim : tuple, default=(0.5, 0.65)
        Global y-axis limits for left axis (e.g., Sample validation)
    right_y_lim : tuple, default=(0.5, 0.65)
        Global y-axis limits for right axis (e.g., Site/Grid validation)
    fit_types : dict, optional
        Global fitting types for all strategies.
    reduce_points : bool, default=False
        If True, uniformly subsample data points to reduce clutter.
    
    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure object
    axes : dict
        Dictionary of axes {'og_mlp': ax1, 'og_resnet': ax2, 'og_transformer': ax3}
    """
    # Set default fit_types if not provided
    if fit_types is None:
        fit_types = {
            'Sample': 'polynomial',
            'Site': 'polynomial',
            'Grid': 'polynomial'
        }
    
    # Create figure and GridSpec
    fig = plt.figure(figsize=figsize)
    gs = gridspec.GridSpec(1, 3, figure=fig, hspace=0.3, wspace=0.6)
    
    # Create axes
    ax_og_mlp = fig.add_subplot(gs[0, 0])          # Left: OG-MLP
    ax_og_resnet = fig.add_subplot(gs[0, 1])       # Middle: OG-ResNet
    ax_og_transformer = fig.add_subplot(gs[0, 2])  # Right: OG-Transformer
    
    confidence_band = (config == 2)
    og_str = "og_" if og else ""

    # Plot OG-MLP (left)
    plot_capacity_strategy_comparison(
        [og_str+'mlp'],
        epoch_project_path,
        assign_train="Sample",
        left_metrics=["Sample"],
        degree=2,
        right_metrics=["Site", "Grid"],
        left_y_lim=left_y_lim,
        right_y_lim=right_y_lim,
        confidence_band=confidence_band,
        fit_types=fit_types,
        exclude_first_n_points=1,
        show_points=True,
        ratio=0.7,
        scale=0.9,
        max_locator=(6, 6),
        ax=ax_og_mlp,
        show_legend=show_legend,
        show_xlabel=xlabel,
        show_title=title,
        show_ylabel=True,
        reduce_points=reduce_points
    )
    
    # Plot OG-ResNet (middle)
    plot_capacity_strategy_comparison(
        [og_str+'resnet'],
        epoch_project_path,
        assign_train="Sample",
        left_metrics=["Sample"],
        right_metrics=["Site", "Grid"],
        left_y_lim=left_y_lim,
        right_y_lim=right_y_lim,
        fit_types=fit_types,
        confidence_band=confidence_band,
        exclude_first_n_points=1,
        ratio=0.8,
        scale=0.9,
        max_locator=(6, 6),
        ax=ax_og_resnet,
        show_legend=False,
        show_xlabel=xlabel,
        show_title=title,
        show_ylabel=True,
        reduce_points=reduce_points
    )
    
    # Plot OG-Transformer (right)
    plot_capacity_strategy_comparison(
        [og_str+'transformer'],
        epoch_project_path,
        assign_train="Sample",
        left_metrics=["Sample"],
        degree=2,
        right_metrics=["Site", "Grid"],
        left_y_lim=left_y_lim,
        right_y_lim=right_y_lim,
        confidence_band=confidence_band,
        fit_types=fit_types,
        exclude_first_n_points=1,
        show_points=True,
        ratio=0.7,
        scale=0.9,
        max_locator=(6, 6),
        ax=ax_og_transformer,
        show_legend=False,
        show_xlabel=xlabel,
        show_title=title,
        show_ylabel=True,
        reduce_points=reduce_points
    )
    
    # Adjust layout
    plt.tight_layout()
    
    # Save if path provided
    if save_path is not None:
        if save_path.lower().endswith('.svg'):
            fig.savefig(save_path, format='svg', bbox_inches='tight')
        else:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Figure 4 (Capacity 1x3) saved to: {save_path}")
    
    # Return figure and axes dictionary
    axes_dict = {
        'og_mlp': ax_og_mlp,
        'og_resnet': ax_og_resnet,
        'og_transformer': ax_og_transformer
    }
    
    return fig, axes_dict


# =============================================================================
# Figure 4a/4b: Single OG-MLP plots (for paper layout)
# =============================================================================

def plot_figure4a(param_project_path="data/appendix/5CV_validation_vs_parameter",
                  figsize=(4.5, 2.3),
                  sample_size='full',
                  save_path=None,
                  use_folds=None,
                  exclude_point_idx=None,
                  y_lim=((0.5, 0.65), (0.5, 0.65)),
                  max_locator=(6, 6),
                  metric="r2_score"):
    """
    Plot Figure 4a: OG-MLP parameter complexity (single subplot).
    
    Parameters
    ----------
    param_project_path : str
        Path to parameter space analysis results
    figsize : tuple
        Figure size (width, height), default (4.5, 2.3) = 1/3 of original 1x3
    sample_size : str or int
        Sample size to analyze ('full' or specific number)
    save_path : str, optional
        Path to save the figure
    use_folds : list of int, optional
        Which folds to include when aggregating
    exclude_point_idx : list of int, optional
        Indices to exclude from fitting
    y_lim : tuple of (left_lim, right_lim)
        Y-axis limits; each lim is (min, max)
    max_locator : tuple
        Max ticks for (left, right) y-axes
    metric : str
        Metric to plot
    
    Returns
    -------
    fig, ax : figure and axis
    """
    left_lim, right_lim = y_lim
    
    fig, ax = plt.subplots(figsize=figsize)
    
    plot_represent_parameter_vs_test_error(
        ['og_mlp'],
        param_project_path,
        x_axis='parameter_count',
        sample_size=sample_size,
        left_metrics=["Sample"],
        right_metrics=["Site", "Grid"],
        degree=2,
        uncertainty='band',
        use_folds=use_folds,
        exclude_point_idx=exclude_point_idx,
        left_y_lim=[left_lim],
        right_y_lim=[right_lim],
        ratio=0.7,
        fit_types={
            'Sample': 'polynomial',
            'Site': 'polynomial',
            'Grid': 'polynomial'
        },
        max_locator=max_locator,
        ax=ax,
        show_legend=False,
        show_title=False,
        show_xlabel=False,
        show_ylabel=True,
        metric=metric
    )
    
    plt.tight_layout()
    
    if save_path is not None:
        if save_path.lower().endswith('.svg'):
            fig.savefig(save_path, format='svg', bbox_inches='tight')
        else:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Figure 4a saved to: {save_path}")
    
    return fig, ax


def plot_figure4b(epoch_project_path="data/appendix/5-Fold-nn_grid/epoch_analysis",
                  figsize=(4.5, 2.3),
                  config=2,
                  save_path=None,
                  left_y_lim=(0.5, 0.65),
                  right_y_lim=(0.5, 0.65),
                  reduce_points=True):
    """
    Plot Figure 4b: OG-MLP training epochs (single subplot).
    
    Parameters
    ----------
    epoch_project_path : str
        Path to epoch training curves results
    figsize : tuple
        Figure size (width, height), default (4.5, 2.3) = 1/3 of original 1x3
    config : int
        Configuration: 1 (no confidence band) or 2 (with confidence band)
    save_path : str, optional
        Path to save the figure
    left_y_lim : tuple
        Y-axis limits for left axis (min, max)
    right_y_lim : tuple
        Y-axis limits for right axis (min, max)
    reduce_points : bool
        If True, uniformly subsample data points
    
    Returns
    -------
    fig, ax : figure and axis
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    confidence_band = (config == 2)
    
    plot_capacity_strategy_comparison(
        ['og_mlp'],
        epoch_project_path,
        assign_train="Sample",
        left_metrics=["Sample"],
        degree=2,
        right_metrics=["Site", "Grid"],
        left_y_lim=left_y_lim,
        right_y_lim=right_y_lim,
        confidence_band=confidence_band,
        fit_types={
            'Sample': 'polynomial',
            'Site': 'polynomial',
            'Grid': 'polynomial'
        },
        exclude_first_n_points=1,
        show_points=True,
        ratio=0.7,
        scale=0.9,
        max_locator=(6, 6),
        ax=ax,
        show_legend=False,
        show_xlabel=False,
        show_title=False,
        show_ylabel=True,
        reduce_points=reduce_points
    )
    
    plt.tight_layout()
    
    if save_path is not None:
        if save_path.lower().endswith('.svg'):
            fig.savefig(save_path, format='svg', bbox_inches='tight')
        else:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Figure 4b saved to: {save_path}")
    
    return fig, ax
