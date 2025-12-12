"""
Surface fitting utilities for accuracy-sparsity analysis.

Core functions for two_stage model:
- TwoStageModel: GAM (Density) + SVM (Spatial Residual)
- fit_universal_models: Fit models for accuracy prediction
- prepare_spatial_features: Aggregate data by spatial grid
- find_bins_intervals: Find bin boundaries for sparsity/sufficiency

Dependencies: sklearn, pygam, numpy, pandas
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.metrics import r2_score
from scipy.stats import pearsonr
import warnings
warnings.filterwarnings('ignore')


# =============================================================================
# Bin Intervals
# =============================================================================

def find_bins_intervals(df, sparsity_bins):
    """
    Find bin boundaries on global df.
    
    Returns:
        sparsity_bins_edges, sufficiency_to_bin_mapping
    """
    # Sparsity bin edges
    sparsity_cut_result = pd.cut(df['sparsity'], bins=sparsity_bins, labels=False, retbins=True)
    sparsity_bins_edges = sparsity_cut_result[1]
    
    # Sufficiency bin mapping
    unique_suff_values = np.sort(df['sufficiency'].unique())
    n_unique = len(unique_suff_values)
    sufficiency_bins_num = n_unique
    values_per_bin = n_unique // sufficiency_bins_num
    remainder = n_unique % sufficiency_bins_num
    
    suff_to_bin = {}
    current_idx = 0
    for bin_idx in range(sufficiency_bins_num):
        current_bin_size = values_per_bin + (1 if bin_idx < remainder else 0)
        for i in range(current_bin_size):
            if current_idx < n_unique:
                suff_to_bin[unique_suff_values[current_idx]] = bin_idx
                current_idx += 1
    
    return sparsity_bins_edges, suff_to_bin


# =============================================================================
# Data Preparation
# =============================================================================

def prepare_spatial_features(df, model_names, split_by='grid', include_sampling_features=True, 
                             bins_intervals=None, resolution=None, outlier_threshold=None, clip=None):
    """
    Prepare data: aggregate by spatial grid with optional sampling features.
    
    Args:
        df: Input data
        model_names: List of model names
        split_by: 'grid' or 'station'
        include_sampling_features: Whether to include sufficiency_log and sparsity
        bins_intervals: If include_sampling_features=True, provide bin boundaries
        resolution: [lon_bins, lat_bins], default [10, 10]
        clip: R² clipping range [min, max]
    
    Returns:
        {model_name: DataFrame with columns [longitude, latitude, (sparsity, sufficiency_log), r2]}
    """
    if resolution is None:
        resolution = [10, 10]
    
    df_copy = df.copy()
    
    if split_by == 'grid':
        df_copy['longitude_bin'] = pd.cut(df_copy['longitude'], bins=resolution[0], labels=False)
        df_copy['latitude_bin'] = pd.cut(df_copy['latitude'], bins=resolution[1], labels=False)
        group_cols = ['longitude_bin', 'latitude_bin']
    else:  # station
        if 'Site_number' in df_copy.columns:
            df_copy['station_id'] = df_copy['Site_number']
        else:
            df_copy['station_id'] = df_copy.groupby(['longitude', 'latitude']).ngroup()
        group_cols = ['station_id']
    
    df_copy = df_copy.dropna(subset=group_cols)
    
    # If need sampling features, assign bins
    if include_sampling_features and bins_intervals is not None:
        sparsity_bins_edges, suff_to_bin = bins_intervals
        df_copy['sparsity_bin'] = pd.cut(df_copy['sparsity'], bins=sparsity_bins_edges, labels=False)
        df_copy['sufficiency_bin'] = df_copy['sufficiency'].map(suff_to_bin)
        group_cols = group_cols + ['sufficiency_bin']
    
    results_dict = {}
    
    for model_name in model_names:
        model_col = f'predicted_{model_name}'
        if model_col not in df_copy.columns:
            continue
        
        results = []
        
        for name, group in df_copy.groupby(group_cols):
            if len(group) < 3:
                continue
            
            observed = group['observed'].values
            predicted = group[model_col].values
            valid_mask = ~(np.isnan(observed) | np.isnan(predicted) | 
                          np.isinf(observed) | np.isinf(predicted))
            
            if valid_mask.sum() < 5:
                continue
            
            r2 = r2_score(observed[valid_mask], predicted[valid_mask])
            
            # Clip R² values
            if clip is not None and len(clip) == 2:
                r2 = np.clip(r2, clip[0], clip[1])
            
            result_row = {
                'longitude': group['longitude'].mean(),
                'latitude': group['latitude'].mean(),
                'r2': r2,
                'count': len(group)
            }
            
            if include_sampling_features and bins_intervals is not None:
                result_row['sparsity'] = group['sparsity'].mean()
                result_row['sufficiency_log'] = np.log10(group['sufficiency'].mean())
            
            results.append(result_row)
        
        if results:
            results_dict[model_name] = pd.DataFrame(results)
    
    return results_dict


def prepare_sampling_features_space(bins_intervals, df, model_names, split_by='grid', 
                                    resolution=None, use_seeds=False, clip=None):
    """
    Prepare training data with sampling features for TwoStageModel Stage 1.
    
    Returns:
        {model_name: DataFrame with [longitude, latitude, sufficiency_log, sparsity, r2]}
    """
    if resolution is None:
        resolution = [10, 10]
    
    sparsity_bins_edges, suff_to_bin = bins_intervals
    df_copy = df.copy()
    
    # Add bins
    df_copy['sparsity_bin'] = pd.cut(df_copy['sparsity'], bins=sparsity_bins_edges, labels=False)
    df_copy['sufficiency_bin'] = df_copy['sufficiency'].map(suff_to_bin)
    
    if split_by == 'grid':
        df_copy['longitude_bin'] = pd.cut(df_copy['longitude'], bins=resolution[0], labels=False)
        df_copy['latitude_bin'] = pd.cut(df_copy['latitude'], bins=resolution[1], labels=False)
        group_cols = ['longitude_bin', 'latitude_bin', 'sufficiency_bin']
    else:
        if 'Site_number' in df_copy.columns:
            df_copy['station_id'] = df_copy['Site_number']
        else:
            df_copy['station_id'] = df_copy.groupby(['longitude', 'latitude']).ngroup()
        group_cols = ['station_id', 'sufficiency_bin']
    
    df_copy = df_copy.dropna(subset=group_cols + ['sparsity_bin'])
    
    results_dict = {}
    
    for model_name in model_names:
        model_col = f'predicted_{model_name}'
        if model_col not in df_copy.columns:
            continue
        
        results = []
        
        for name, group in df_copy.groupby(group_cols):
            if len(group) < 3:
                continue
            
            observed = group['observed'].values
            predicted = group[model_col].values
            valid_mask = ~(np.isnan(observed) | np.isnan(predicted))
            
            if valid_mask.sum() < 5:
                continue
            
            r2 = r2_score(observed[valid_mask], predicted[valid_mask])
            
            if clip is not None and len(clip) == 2:
                r2 = np.clip(r2, clip[0], clip[1])
            
            results.append({
                'longitude': group['longitude'].mean(),
                'latitude': group['latitude'].mean(),
                'sufficiency_log': np.log10(group['sufficiency'].mean()),
                'sparsity': group['sparsity'].mean(),
                'r2': r2,
                'count': len(group)
            })
        
        if results:
            results_dict[model_name] = pd.DataFrame(results)
    
    return results_dict


# =============================================================================
# TwoStageModel: GAM (Density) + SVM (Spatial Residual)
# =============================================================================

class TwoStageModel:
    """
    Two-stage model: GAM (Density) + SVM (Spatial Residual)
    
    Stage 1: Use sampling aggregation + GAM_Monotonic to predict base R² (density features)
    Stage 2: Use spatial aggregation + SVM to predict residual (spatial features)
    Final: Prediction = GAM prediction + SVM residual prediction
    """
    
    def __init__(self, spline=7, lam=0.5, resolution=None, spline_order=2, clip=None):
        """
        Args:
            spline: Number of GAM spline knots
            lam: GAM regularization parameter
            resolution: Spatial aggregation resolution [lon_bins, lat_bins]
            spline_order: Spline order (0=constant, 1=linear, 2=quadratic, 3=cubic)
            clip: R² clipping range [min, max], default [-0.5, 1.0]
        """
        self.spline = spline
        self.lam = lam
        self.resolution = resolution if resolution else [10, 10]
        self.spline_order = spline_order
        self.clip = clip if clip is not None else [-0.5, 1.0]
        
        # Stage 1: GAM model (Density features)
        self.gam_model = None
        self.gam_scaler = None
        
        # Stage 2: SVM model (Spatial features for residual)
        self.svm_model = None
        self.svm_scaler = None
        
        # Auto-detect: single sufficiency value
        self.single_sufficiency = False
        self.stage2_features = ['longitude', 'latitude', 'sufficiency_log']
    
    def fit(self, df_train_raw, model_name, bins_intervals, use_seeds=True, split_by='grid', metric='r2'):
        """
        Train two-stage model.
        
        Returns:
            stage1_score: Stage 1 GAM score
            stage2_score: Stage 2 (GAM + SVM) score
        """
        from pygam import LinearGAM, s, te
        
        # ====== Stage 1: Train GAM with sampling aggregation ======
        stage1_data_dict = prepare_sampling_features_space(
            bins_intervals, df_train_raw, [model_name], 
            split_by=split_by,
            resolution=self.resolution,
            use_seeds=use_seeds,
            clip=self.clip
        )
        
        stage1_data = stage1_data_dict.get(model_name)
        
        if stage1_data is None or len(stage1_data) < 5:
            raise ValueError(f"Insufficient data for model {model_name} in stage 1")
        
        # Auto-detect: single sufficiency value
        suff_log_values = stage1_data['sufficiency_log'].values
        suff_log_std = np.std(suff_log_values)
        self.single_sufficiency = (suff_log_std < 1e-6)
        
        # Extract density features
        if self.single_sufficiency:
            X_stage1 = stage1_data[['sparsity']].values
        else:
            X_stage1 = stage1_data[['sufficiency_log', 'sparsity']].values
        
        y_stage1 = stage1_data['r2'].values
        
        # Train GAM with monotonic constraints
        self.gam_scaler = StandardScaler()
        X_stage1_scaled = self.gam_scaler.fit_transform(X_stage1)
        
        adjusted_spline = self.spline
        
        if self.single_sufficiency:
            # Single variable GAM: only sparsity (monotonic increasing)
            self.gam_model = LinearGAM(
                s(0, constraints='monotonic_inc', n_splines=adjusted_spline, 
                  spline_order=self.spline_order, lam=self.lam)
            )
        else:
            # Two variable GAM: sufficiency_log + sparsity + interaction
            interaction_spline = max(self.spline_order + 1, adjusted_spline // 4)
            self.gam_model = LinearGAM(
                s(0, constraints='monotonic_inc', n_splines=adjusted_spline, 
                  spline_order=self.spline_order, lam=self.lam) + 
                s(1, constraints='monotonic_inc', n_splines=adjusted_spline, 
                  spline_order=self.spline_order, lam=self.lam) +
                te(0, 1, n_splines=[interaction_spline, interaction_spline], 
                   spline_order=self.spline_order, lam=self.lam*2)
            )
        
        self.gam_model.fit(X_stage1_scaled, y_stage1)
        
        # Evaluate Stage 1
        y_gam_pred_train = self.gam_model.predict(X_stage1_scaled)
        if metric == 'correlation':
            stage1_score, _ = pearsonr(y_stage1, y_gam_pred_train)
        else:
            stage1_score = r2_score(y_stage1, y_gam_pred_train)
        
        # ====== Stage 2: Train SVM with spatial aggregation (predict residual) ======
        stage2_data_dict = prepare_spatial_features(
            df_train_raw, [model_name],
            split_by=split_by,
            include_sampling_features=True,
            bins_intervals=bins_intervals,
            resolution=self.resolution,
            clip=self.clip
        )
        stage2_data = stage2_data_dict.get(model_name)
        
        if stage2_data is None or len(stage2_data) < 5:
            raise ValueError(f"Insufficient spatial data for model {model_name} in stage 2")
        
        # Calculate residual
        y_true = stage2_data['r2'].values
        
        if self.single_sufficiency:
            X_stage2_density = stage2_data[['sparsity']].values
        else:
            X_stage2_density = stage2_data[['sufficiency_log', 'sparsity']].values
        
        X_stage2_density_scaled = self.gam_scaler.transform(X_stage2_density)
        y_gam_pred = self.gam_model.predict(X_stage2_density_scaled)
        residuals = y_true - y_gam_pred
        
        # Train SVM on residuals
        X_stage2_spatial = stage2_data[self.stage2_features].values
        self.svm_scaler = StandardScaler()
        X_stage2_spatial_scaled = self.svm_scaler.fit_transform(X_stage2_spatial)
        
        self.svm_model = SVR(kernel='rbf', C=1.0, gamma='scale')
        self.svm_model.fit(X_stage2_spatial_scaled, residuals)
        
        # Evaluate Stage 2
        y_svm_pred = self.svm_model.predict(X_stage2_spatial_scaled)
        y_final_pred = y_gam_pred + y_svm_pred
        
        if metric == 'correlation':
            stage2_score, _ = pearsonr(y_true, y_final_pred)
        else:
            stage2_score = r2_score(y_true, y_final_pred)
        
        return stage1_score, stage2_score
    
    def predict(self, X_spatial, sparsity_values, sufficiency_log_values=None):
        """
        Predict R² for new data points.
        
        Args:
            X_spatial: Spatial features [longitude, latitude]
            sparsity_values: Sparsity values
            sufficiency_log_values: Log10(sufficiency) values (optional if single_sufficiency)
        
        Returns:
            Predicted R² values
        """
        if self.gam_model is None or self.svm_model is None:
            raise ValueError("Model not fitted. Call fit() first.")
        
        # Prepare density features
        if self.single_sufficiency:
            X_density = np.array(sparsity_values).reshape(-1, 1)
        else:
            if sufficiency_log_values is None:
                raise ValueError("sufficiency_log_values required for multi-sufficiency model")
            X_density = np.column_stack([sufficiency_log_values, sparsity_values])
        
        # Stage 1: GAM prediction
        X_density_scaled = self.gam_scaler.transform(X_density)
        y_gam = self.gam_model.predict(X_density_scaled)
        
        # Stage 2: SVM residual prediction
        X_spatial_scaled = self.svm_scaler.transform(X_spatial)
        y_svm = self.svm_model.predict(X_spatial_scaled)
        
        return y_gam + y_svm


# =============================================================================
# Universal Model Fitting
# =============================================================================

def fit_universal_models(df, sparsity_bins, model_type='two_stage', verbose=True,
                         full_features='Full', resolution=None, splitby='grid',
                         spline=7, lam=0.5, spline_order=2):
    """
    Fit models for accuracy prediction.
    
    Args:
        df: DataFrame with observed, predicted_*, longitude, latitude, sparsity, sufficiency
        sparsity_bins: Number of sparsity bins
        model_type: 'two_stage' (GAM + SVM)
        verbose: Print progress
        full_features: 'Full' or list of features
        resolution: [lon_bins, lat_bins]
        splitby: 'grid' or 'station'
        spline: GAM spline knots
        lam: GAM regularization
        spline_order: GAM spline order
    
    Returns:
        dict: {model_name: {'model': TwoStageModel, 'r2_score': float, ...}}
    """
    if resolution is None:
        resolution = [10, 10]
    
    # Get model names
    model_columns = [col for col in df.columns if col.startswith('predicted_')]
    model_names = [col.replace('predicted_', '') for col in model_columns]
    
    bins_intervals = find_bins_intervals(df, sparsity_bins)
    
    fitted_models = {}
    
    for model_name in model_names:
        if verbose:
            print(f"Fitting {model_type} model for {model_name}...")
        
        try:
            two_stage_model = TwoStageModel(
                spline=spline, lam=lam, resolution=resolution, 
                spline_order=spline_order
            )
            stage1_score, stage2_score = two_stage_model.fit(
                df, model_name, bins_intervals,
                use_seeds=True, split_by=splitby, metric='r2'
            )
            
            fitted_models[model_name] = {
                'model': two_stage_model,
                'scaler': None,
                'r2_score': stage2_score,
                'model_type': model_type,
                'full_features': full_features,
                'stage1_score': stage1_score,
                'stage2_score': stage2_score
            }
            
            if verbose:
                print(f"  {model_name}: Stage1={stage1_score:.4f}, Stage2={stage2_score:.4f}")
        
        except Exception as e:
            if verbose:
                print(f"  {model_name}: Failed - {e}")
    
    return fitted_models

