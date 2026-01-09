"""
Heatmap visualization for accuracy surface analysis (Figure 5e).

This module provides functions to create heatmap comparisons between
observed R² values and GAM-predicted R² values across different
sufficiency and sparsity levels.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import gridspec


def plot_heatmap_comparison(fitted_models, df, model_name, sparsity_bins=10,
                            color_range=[0.2, 0.76], cmap='Spectral_r', 
                            square_size=6, add_text=True, fontsize=10, 
                            tick_fontsize=10, save_path=None, 
                            heatmap_by='spatial_sampling', split_by='grid', 
                            resolution=None, use_seeds=True):
    """
    Plot heatmap comparison: Observed R² vs GAM Predicted R².
    
    This function creates a side-by-side visualization showing:
    - Left panel: Observed R² from actual data aggregation
    - Right panel: GAM-predicted R² from the fitted model
    
    Parameters
    ----------
    fitted_models : dict
        Dictionary from fit_universal_models() containing model info
    df : pd.DataFrame
        Data with columns: observed, predicted_*, longitude, latitude, 
        sparsity, sufficiency
    model_name : str
        Model name (e.g., 'LightGBM', 'MLP_OG')
    sparsity_bins : int, default=10
        Number of sparsity bins for aggregation
    color_range : list, default=[0.2, 0.76]
        [vmin, vmax] for colorbar range
    cmap : str, default='Spectral_r'
        Matplotlib colormap name
    square_size : float, default=6
        Size of each heatmap subplot in inches
    add_text : bool, default=True
        Whether to add text annotations showing R² values
    fontsize : int, default=10
        Font size for R² value annotations
    tick_fontsize : int, default=10
        Font size for axis tick labels
    save_path : str or None, default=None
        Path to save figure (e.g., 'figure5e.svg')
    heatmap_by : str, default='spatial_sampling'
        Aggregation method:
        - 'spatial_sampling': Spatial aggregation followed by density binning
        - 'sampling': Direct density binning (not recommended)
    split_by : str, default='grid'
        Spatial splitting method: 'grid' or 'station'
    resolution : list or None, default=None
        Spatial grid resolution [lon_bins, lat_bins], default [30, 30]
    use_seeds : bool, default=True
        Whether to use random seeds in aggregation
    
    Returns
    -------
    observed_matrix : np.ndarray
        2D array of observed R² values (m x n)
    predicted_matrix : np.ndarray
        2D array of GAM-predicted R² values (m x n)
    
    Notes
    -----
    The function uses spatial→sampling aggregation by default, which:
    1. First aggregates data spatially (by grid cells or stations)
    2. Then bins by sufficiency and sparsity levels
    This approach better captures spatial patterns than direct binning.
    """
    from .fit_surface import find_bins_intervals, prepare_sampling_features_space
    
    if model_name not in fitted_models:
        print(f"❌ Model {model_name} not found in fitted_models!")
        print(f"Available models: {list(fitted_models.keys())}")
        return None, None
    
    if resolution is None:
        resolution = [30, 30]
    
    # Prepare bin intervals using the sparsity_bins parameter
    # This is separate from the bins used in GAM training!
    bins_intervals = find_bins_intervals(df, sparsity_bins)
    
    # Aggregate data using spatial→sampling aggregation
    model_data_dict = prepare_sampling_features_space(
        bins_intervals, df, [model_name],
        split_by=split_by,
        resolution=resolution,
        use_seeds=use_seeds
    )
    
    model_data = model_data_dict.get(model_name)
    
    if model_data is None or len(model_data) == 0:
        print(f"❌ No data found for model {model_name}")
        return None, None
    
    # Check if bin columns exist
    if 'sufficiency_bin' not in model_data.columns or 'sparsity_bin' not in model_data.columns:
        raise ValueError("model_data must contain 'sufficiency_bin' and 'sparsity_bin' columns")
    
    # Get unique bins (these are already computed correctly by prepare_sampling_features_space)
    unique_suff_bins = np.sort(model_data['sufficiency_bin'].dropna().unique())
    unique_spar_bins = np.sort(model_data['sparsity_bin'].dropna().unique())
    
    m = len(unique_suff_bins)
    n = len(unique_spar_bins)
    
    if m == 0 or n == 0:
        print(f"❌ No valid bins found (m={m}, n={n})")
        return None, None
    
    # Calculate bin representative values
    suff_bin_to_val = {}
    spar_bin_to_val = {}
    
    for suff_bin in unique_suff_bins:
        bin_data = model_data[model_data['sufficiency_bin'] == suff_bin]
        suff_bin_to_val[suff_bin] = 10 ** bin_data['sufficiency_log'].mean()
    
    for spar_bin in unique_spar_bins:
        bin_data = model_data[model_data['sparsity_bin'] == spar_bin]
        spar_bin_to_val[spar_bin] = bin_data['sparsity'].mean()
    
    unique_sufficiency = np.array([suff_bin_to_val[b] for b in unique_suff_bins])
    unique_sparsity = np.array([spar_bin_to_val[b] for b in unique_spar_bins])
    
    # Create observed matrix
    observed_matrix = np.full((m, n), np.nan)
    
    for _, row in model_data.iterrows():
        suff_bin = row['sufficiency_bin']
        spar_bin = row['sparsity_bin']
        
        if pd.isna(suff_bin) or pd.isna(spar_bin):
            continue
        
        i = np.where(unique_suff_bins == suff_bin)[0]
        j = np.where(unique_spar_bins == spar_bin)[0]
        
        if len(i) > 0 and len(j) > 0:
            observed_matrix[i[0], j[0]] = row['r2']
    
    # Get fitted model
    model_info = fitted_models[model_name]
    two_stage_model = model_info['model']
    
    # Create predicted matrix using Stage 1 GAM only
    predicted_matrix = np.full((m, n), np.nan)
    
    for i, suff_val in enumerate(unique_sufficiency):
        for j, spar_val in enumerate(unique_sparsity):
            # Prepare density features (same as GAM training)
            X_pred = np.array([[np.log10(suff_val), spar_val]])
            
            # Use Stage 1 only (GAM prediction without spatial residual)
            try:
                predicted_matrix[i, j] = two_stage_model.predict_stage1_only(X_pred)[0]
            except Exception as e:
                # Silent failure for individual predictions
                continue
    
    # ===== Plotting =====
    colorbar_width = 0.5  # colorbar width
    spacing = 5.0  # spacing between subplots (increased spacing)
    total_width = 2 * square_size + colorbar_width + spacing
    figsize = (total_width, square_size)
    
    fig = plt.figure(figsize=figsize)
    gs = gridspec.GridSpec(
        1, 3, 
        width_ratios=[square_size, square_size, colorbar_width],
        figure=fig, 
        wspace=0.15
    )
    
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    cax = fig.add_subplot(gs[0, 2])
    
    # Format labels
    def format_sufficiency_label(value):
        """Format sufficiency value as k/m notation"""
        if value >= 1000000:
            return f"{int(value/1000000)}m"
        elif value >= 1000:
            return f"{int(value/1000)}k"
        else:
            return f"{int(value)}"
    
    def format_sparsity_label(value):
        """Format sparsity value in scientific notation"""
        return f"{value:.1e}"
    
    # Create tick labels (show every ~5th tick)
    suff_tick_idx = range(0, m, max(1, m//5))
    spar_tick_idx = range(0, n, max(1, n//5))
    
    suff_labels = [format_sufficiency_label(unique_sufficiency[i]) for i in suff_tick_idx]
    spar_labels = [format_sparsity_label(unique_sparsity[i]) for i in spar_tick_idx]
    
    vmin, vmax = color_range
    
    # Left panel: Observed R²
    aggregation_label = "Sampling" if heatmap_by == 'sampling' else "Spatial→Sampling"
    im1 = ax1.imshow(
        observed_matrix, 
        cmap=cmap, 
        aspect='equal', 
        origin='lower', 
        vmin=vmin, 
        vmax=vmax
    )
    ax1.set_title(f'{model_name} - Observed R² ({aggregation_label})', fontsize=12)
    # ax1.set_xlabel('Sparsity Bins', fontsize=tick_fontsize)
    # ax1.set_ylabel('Sufficiency Values', fontsize=tick_fontsize)
    ax1.set_xticks(spar_tick_idx)
    ax1.set_xticklabels(spar_labels, fontsize=tick_fontsize)
    ax1.set_yticks(suff_tick_idx)
    ax1.set_yticklabels(suff_labels, fontsize=tick_fontsize)
    
    # Add text annotations to left panel
    if add_text:
        for i in range(m):
            for j in range(n):
                if not np.isnan(observed_matrix[i, j]):
                    ax1.text(
                        j, i, f'{observed_matrix[i, j]:.2f}',
                        ha='center', va='center', 
                        color='white', fontsize=fontsize
                    )
    
    # Right panel: Predicted R²
    im2 = ax2.imshow(
        predicted_matrix, 
        cmap=cmap, 
        aspect='equal', 
        origin='lower', 
        vmin=vmin, 
        vmax=vmax
    )
    ax2.set_title(f'{model_name} - GAM Predicted R²', fontsize=12)
    ax2.set_xlabel('Density Level', fontsize=tick_fontsize)
    ax2.set_ylabel('Sufficiency Values', fontsize=tick_fontsize)
    ax2.set_xticks(spar_tick_idx)
    ax2.set_xticklabels(spar_labels, fontsize=tick_fontsize)
    ax2.set_yticks(suff_tick_idx)
    ax2.set_yticklabels(suff_labels, fontsize=tick_fontsize)
    
    # Add text annotations to right panel
    if add_text:
        for i in range(m):
            for j in range(n):
                if not np.isnan(predicted_matrix[i, j]):
                    ax2.text(
                        j, i, f'{predicted_matrix[i, j]:.2f}',
                        ha='center', va='center', 
                        color='white', fontsize=fontsize
                    )
    
    # Colorbar
    cbar = plt.colorbar(im1, cax=cax, label='R²')
    cbar.ax.tick_params(labelsize=tick_fontsize)
    
    plt.tight_layout()
    
    # Save figure if path provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Figure saved to: {save_path}")
    
    plt.show()
    
    # Print statistics
    obs_valid = observed_matrix[~np.isnan(observed_matrix)]
    pred_valid = predicted_matrix[~np.isnan(predicted_matrix)]
    
    print(f"\n{'='*60}")
    print(f"Heatmap Statistics for {model_name}")
    print(f"{'='*60}")
    print(f"Matrix shape: {m} x {n} (sufficiency x sparsity)")
    print(f"Observed R² range: [{obs_valid.min():.3f}, {obs_valid.max():.3f}]")
    print(f"Predicted R² range: [{pred_valid.min():.3f}, {pred_valid.max():.3f}]")
    print(f"Valid cells: {len(obs_valid)}/{m*n} ({100*len(obs_valid)/(m*n):.1f}%)")
    print(f"{'='*60}\n")
    
    return observed_matrix, predicted_matrix

