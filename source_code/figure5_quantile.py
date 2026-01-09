"""
Quartile Analysis for Sparsity-based R² Evaluation

Calculate and compare model R² across different sparsity quartiles.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score


def calculate_quartile_r2(df, models=None, sparsity_col='sparsity', observed_col='observed', 
                         quartiles_count=4, levels=None, prefix='predicted_', print_results=True):
    """
    Calculate model R² statistics across different sparsity quartile intervals.
    
    Args:
        df: DataFrame containing the data
        models: list of model names to analyze. If None, will auto-detect from columns
        sparsity_col: name of the sparsity column
        observed_col: name of the observed values column
        quartiles_count: number of quartiles to divide the data (default: 4, ignored if levels is provided)
        levels: custom sparsity levels to define boundaries. If provided, quartiles_count is ignored
        prefix: prefix for predicted columns (default: 'predicted_')
        print_results: whether to print results to console
    
    Returns:
        results_dict: dictionary with R² values for each model and quartile
        summary_df: DataFrame with formatted results (Quartile, Number, model1, model2, ...)
    """
    
    # 1. Auto-detect models if not provided
    if models is None:
        predicted_cols = [col for col in df.columns if col.startswith(prefix)]
        models = [col.replace(prefix, '') for col in predicted_cols]
        if print_results:
            print(f"Auto-detected models: {models}")
    
    # 2. Calculate quartile boundaries
    if levels is not None:
        # Use custom levels
        custom_levels = np.array(levels)
        # Sort in ascending order for pd.cut
        custom_levels_sorted = np.sort(custom_levels)
        
        # Add min and max boundaries if needed
        min_sparsity = df[sparsity_col].min()
        max_sparsity = df[sparsity_col].max()
        
        # Ensure boundaries cover all data
        boundaries = []
        if custom_levels_sorted[0] > min_sparsity:
            boundaries.append(min_sparsity - 1e-15)  # Slightly below minimum
        boundaries.extend(custom_levels_sorted)
        if custom_levels_sorted[-1] < max_sparsity:
            boundaries.append(max_sparsity + 1e-15)  # Slightly above maximum
        
        quartile_values = np.array(boundaries)
        quartiles_count = len(quartile_values) - 1
        
        # Store original levels for boundary display (descending order)
        display_levels = np.sort(custom_levels)[::-1]
    else:
        # Use percentile-based quartiles
        quartile_percentiles = np.linspace(0, 100, quartiles_count + 1)
        quartile_values = np.percentile(df[sparsity_col], quartile_percentiles)
        display_levels = quartile_values[::-1]  # For display purposes
    
    # 3. Create quartile labels
    quartile_labels = [f'Q{i}' for i in range(quartiles_count)]
    
    # 4. Assign quartile groups to each data point
    df_copy = df.copy()
    
    # quartile_values is already in ascending order for pd.cut
    # But we want Q0 to represent the highest sparsity range
    # So we reverse the labels
    labels_reversed = quartile_labels[::-1]
    
    df_copy['quartile'] = pd.cut(
        df_copy[sparsity_col], 
        bins=quartile_values, 
        labels=labels_reversed, 
        include_lowest=True
    )
    
    # 5. Calculate R² for each model and quartile
    results = {}
    data_counts = {}
    
    for model in models:
        model_r2 = {}
        model_counts = {}
        predicted_col = f'{prefix}{model}'
        
        if predicted_col not in df_copy.columns:
            if print_results:
                print(f"Warning: Column {predicted_col} not found. Skipping {model}")
            continue
            
        for quartile in quartile_labels:
            # Get data for this quartile
            mask = df_copy['quartile'] == quartile
            quartile_data = df_copy[mask]
            
            if len(quartile_data) > 0:
                # Get observed and predicted values
                observed = quartile_data[observed_col]
                predicted = quartile_data[predicted_col]
                
                # Remove NaN values
                valid_mask = ~(np.isnan(observed) | np.isnan(predicted))
                valid_observed = observed[valid_mask]
                valid_predicted = predicted[valid_mask]
                
                model_counts[quartile] = len(valid_observed)
                
                if len(valid_observed) > 1:
                    r2 = r2_score(valid_observed, valid_predicted)
                    model_r2[quartile] = r2
                else:
                    model_r2[quartile] = np.nan
            else:
                model_r2[quartile] = np.nan
                model_counts[quartile] = 0
        
        results[model] = model_r2
        data_counts[model] = model_counts
    
    # 6. Create summary DataFrame in the requested format
    summary_data = []
    
    for i, quartile in enumerate(quartile_labels):
        # Fix: reverse the boundary index to match reversed labels
        boundary_idx = len(quartile_labels) - 1 - i
        row_data = {
            'Quartile': quartile,
            'Boundary': f"{quartile_values[boundary_idx]:.2e} - {quartile_values[boundary_idx+1]:.2e}",
            'Number': data_counts[models[0]][quartile] if models and models[0] in data_counts else 0
        }
        
        # Add R² values for each model
        for model in models:
            if model in results:
                r2_val = results[model][quartile]
                if not np.isnan(r2_val):
                    row_data[model] = f"{r2_val:.4f}"
                else:
                    row_data[model] = "N/A"
            else:
                row_data[model] = "N/A"
        
        summary_data.append(row_data)
    
    summary_df = pd.DataFrame(summary_data)
    
    # 7. Print results if requested
    if print_results:
        print(f"\nSparsity Quartile Boundaries:")
        for i, quartile in enumerate(quartile_labels):
            # Fix: reverse the boundary index to match reversed labels
            boundary_idx = len(quartile_labels) - 1 - i
            print(f"  {quartile}: {quartile_values[boundary_idx]:.2e} - {quartile_values[boundary_idx+1]:.2e}")
            
        print(f"\nSummary Table:")
        print(summary_df.to_string(index=False))
    
    return results, summary_df


def compare_models_quartile_r2(df, models=['LightGBM', 'Ours'], **kwargs):
    """
    Convenience function: compare specified models' R² performance across quartile intervals.
    
    Args:
        df: DataFrame containing the data
        models: list of models to compare (default: ['LightGBM', 'Ours'])
        **kwargs: additional arguments passed to calculate_quartile_r2
    
    Returns:
        results_dict: dictionary with R² values
        results_df: DataFrame with R² values
        comparison_df: DataFrame showing model differences
    """
    
    results, summary_df = calculate_quartile_r2(df, models=models, **kwargs)
    
    # Calculate differences if exactly 2 models
    if len(models) == 2:
        model1, model2 = models
        if model1 in results and model2 in results:
            differences = {}
            quartile_labels = [row['Quartile'] for _, row in summary_df.iterrows()]
            for quartile in quartile_labels:
                r2_1 = results[model1][quartile]
                r2_2 = results[model2][quartile]
                if not (np.isnan(r2_1) or np.isnan(r2_2)):
                    differences[quartile] = r2_2 - r2_1  # model2 - model1
                else:
                    differences[quartile] = np.nan
            
            # Add difference column to summary_df
            summary_df[f'{model2}_minus_{model1}'] = [
                f"{differences[q]:.4f}" if not np.isnan(differences[q]) else "N/A" 
                for q in quartile_labels
            ]
            
            print(f"\nModel Comparison Added ({model2} - {model1})")
            
            return results, summary_df
    
    return results, summary_df

