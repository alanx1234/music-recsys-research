import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from pathlib import Path

# Use LaTeX font rendering if available for a professional look
plt.rcParams.update({
    "font.family": "serif",
    "text.usetex": False, # Set to True if you have TeX installed
    "axes.labelsize": 12,
    "legend.fontsize": 10,
})

def plot_polished_tradeoff(csv_path: Path, out_path: Path):
    df = pd.read_csv(csv_path)
    
    # Filter for a specific K to keep the plot focused
    df_k5 = df[df['K'] == 5].copy()
    
    fig, ax = plt.subplots(figsize=(7, 5), dpi=300)
    
    # Define distinct markers and colors for each variant
    # Shape-Track: df_k5 has N rows corresponding to ranking variants
    style_map = {
        "va_only": {"m": "o", "c": "#377eb8", "label": "VA Only (Baseline)"},
        "va_plus_cte": {"m": "s", "c": "#ff7f00", "label": "VA + Complexity"},
        "va_plus_ngram": {"m": "^", "c": "#4daf4a", "label": "VA + Grammar"},
        "full": {"m": "D", "c": "#e41a1c", "label": "CSRS (Full System)"}
    }

    for variant, style in style_map.items():
        row = df_k5[df_k5['variant'] == variant]
        if row.empty: continue
        
        # Scale bubble size by n-gram similarity for the 3rd dimension
        size = row['mean_ngram_sim'].values[0] * 2000 + 100 
        
        ax.scatter(
            row['mean_gt_va_dist'], 
            row['mean_abs_cte_diff'],
            s=size, 
            c=style['c'], 
            marker=style['m'],
            label=style['label'],
            alpha=0.8,
            edgecolors='black',
            linewidths=0.5
        )

    # Labeling with scientific notation/units
    ax.set_xlabel(r"Mean Ground-Truth $(V,A)$ Distance $\downarrow$ (Lower is Better)")
    ax.set_ylabel(r"Mean $|\Delta \text{CTE}|$ $\downarrow$ (Lower is Better)")
    ax.set_title("Retrieval Performance Tradeoff ($K=5$)", fontsize=14, pad=15)
    
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.legend(frameon=True, loc='upper right')
    
    sns.despine()
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches='tight')
    print(f"Polished figure saved to: {out_path}")