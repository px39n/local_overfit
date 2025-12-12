"""
Figure styles and color schemes for the paper.
"""

import matplotlib.pyplot as plt
import matplotlib as mpl

# =============================================================================
# Color Palettes
# =============================================================================

COLORS = {
    'primary': '#1f77b4',
    'secondary': '#ff7f0e',
    'success': '#2ca02c',
    'danger': '#d62728',
    'warning': '#ffbb78',
    'info': '#17becf',
    'dark': '#2c3e50',
    'light': '#ecf0f1',
}

MODEL_COLORS = {
    # High-Variance Models
    'lightgbm': '#2ecc71',
    'xgboost': '#3498db',
    'catboost': '#9b59b6',
    # Low-Variance Models
    'mlp': '#e74c3c',
    'resnet': '#f39c12',
    'transformer': '#1abc9c',
    # OG Models
    'og_mlp': '#c0392b',
    'og_resnet': '#d35400',
    'og_transformer': '#16a085',
}

VALIDATION_COLORS = {
    'train': '#95a5a6',
    'Sample': '#3498db',
    'Site': '#2ecc71',
    'Grid': '#f39c12',
    'Spatiotemporal_block': '#e74c3c',
}

# =============================================================================
# Figure Style Configuration
# =============================================================================

def setup_figure_style():
    """Setup matplotlib style for publication-quality figures."""
    plt.style.use('seaborn-v0_8-whitegrid')
    
    mpl.rcParams.update({
        'figure.dpi': 150,
        'savefig.dpi': 300,
        'font.family': 'sans-serif',
        'font.size': 10,
        'axes.labelsize': 11,
        'axes.titlesize': 12,
        'xtick.labelsize': 9,
        'ytick.labelsize': 9,
        'legend.fontsize': 9,
        'figure.titlesize': 14,
        'axes.spines.top': False,
        'axes.spines.right': False,
    })

# =============================================================================
# Color Maps
# =============================================================================

def get_accuracy_cmap():
    """Return colormap for accuracy visualization."""
    return 'Spectral_r'

def get_ozone_cmap():
    """Return colormap for ozone concentration."""
    return 'coolwarm'

