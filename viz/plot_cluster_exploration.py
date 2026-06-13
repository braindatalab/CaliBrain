"""
Cluster exploration figures for CaliBrain benchmark results.

Explores clustering patterns and parameter interactions through:
  1. Heatmap: Improvement by (solver × parameter combinations)
  2. Parallel coordinates: Multi-parameter view of top/bottom performers
  3. PCA dimensionality reduction: Natural clustering in metric space
  4. Parameter interaction effects: How pairs of parameters jointly affect outcomes
  5. Ridge plots: Distribution shifts across parameter levels
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from pathlib import Path 
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────
CSV_PATH   = "results/benchmark_results/benchmark_results_20260306_172109.csv"
OUTPUT_DIR = "results/benchmark_results/figures/cluster_exploration"
SAVE_DPI   = 200

# Color palettes
SOLVER_PALETTE = ["#66c2a5", "#fc8d62", "#8da0cb", "#e78ac3"]
DIVERGING_CMAP = "RdYlGn"  # Red (bad) → Yellow → Green (good)

# All improvement metrics to visualize
IMPROVEMENT_METRICS = [
    ("improvement_mean_calibration_error", "Mean Calibration\nError"),
    ("improvement_mean_signed_deviation", "Mean Signed\nDeviation"),
    ("improvement_mean_absolute_deviation", "Mean Absolute\nDeviation"),
    ("improvement_max_underconfidence_deviation", "Max Underconfidence\nDeviation"),
    ("improvement_max_overconfidence_deviation", "Max Overconfidence\nDeviation"),
]

# ─────────────────────────────────────────────────────────────────────────────
def save_fig(fig, path: Path):
    """Save figure with tight layout."""
    fig.savefig(path, dpi=SAVE_DPI, bbox_inches="tight")
    print(f"  ✓  {path}")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 1: Heatmap of improvement by parameter combinations
# ─────────────────────────────────────────────────────────────────────────────
def fig_improvement_heatmap(df: pd.DataFrame, out: Path):
    """
    Show mean improvement for each parameter combination across all 5 metrics.
    Rows = solver × noise_type, Columns = nnz × SNR.
    One subplot per improvement metric.
    """
    print("[1] Improvement heatmap by parameter combinations")
    
    # Create compound grouping keys
    df = df.copy()
    df["solver_noise"] = df["solver"] + " | " + df["noise_type"]
    df["nnz_snr"] = "nnz=" + df["nnz"].astype(str) + " | SNR=" + df["alpha_SNR"].astype(str)
    
    # Create subplot grid: 1 row, 5 columns (one per metric)
    fig, axes = plt.subplots(1, 5, figsize=(24, 8))
    
    for idx, (metric, metric_label) in enumerate(IMPROVEMENT_METRICS):
        ax = axes[idx]
        
        if metric not in df.columns:
            ax.text(0.5, 0.5, f"{metric} not found", ha="center", va="center")
            ax.set_title(metric_label, fontsize=10, fontweight="bold")
            continue
        
        # Pivot table
        pivot = df.pivot_table(
            values=metric,
            index="solver_noise",
            columns="nnz_snr",
            aggfunc="mean"
        )
        
        # Adaptive colorbar limits
        vmin, vmax = pivot.values.min(), pivot.values.max()
        if metric == "improvement_max_overconfidence_deviation":
            # Clip extreme outliers for this metric
            vmin, vmax = max(vmin, -1000), min(vmax, 10000)
        
        # Plot heatmap
        im = ax.imshow(pivot.values, cmap=DIVERGING_CMAP, aspect="auto", 
                      vmin=vmin, vmax=vmax)
        
        # Set ticks
        ax.set_xticks(np.arange(len(pivot.columns)))
        ax.set_yticks(np.arange(len(pivot.index)))
        ax.set_xticklabels(pivot.columns, rotation=45, ha="right", fontsize=7)
        if idx == 0:
            ax.set_yticklabels(pivot.index, fontsize=7)
        else:
            ax.set_yticklabels([])
        
        # Add values as text
        for i in range(len(pivot.index)):
            for j in range(len(pivot.columns)):
                val = pivot.values[i, j]
                if not np.isnan(val):
                    # Auto-determine text color
                    mid = (vmin + vmax) / 2
                    color = "white" if abs(val - mid) > (vmax - vmin) * 0.3 else "black"
                    
                    # Format based on magnitude
                    if abs(val) < 100:
                        text = f"{val:.1f}"
                    else:
                        text = f"{val:.0f}"
                    
                    ax.text(j, i, text, ha="center", va="center",
                           color=color, fontsize=6, fontweight="bold")
        
        # Colorbar
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.ax.tick_params(labelsize=7)
        
        ax.set_xlabel("NNZ × SNR", fontsize=9, fontweight="bold")
        if idx == 0:
            ax.set_ylabel("Solver × Noise", fontsize=9, fontweight="bold")
        ax.set_title(metric_label, fontsize=10, fontweight="bold", pad=10)
    
    fig.suptitle(
        "Calibration Improvement Heatmaps: All Metrics\n"
        "Which parameter combinations yield best improvement?",
        fontsize=14, fontweight="bold", y=0.98
    )
    
    fig.tight_layout()
    save_fig(fig, out / "1_improvement_heatmap.png")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 2: Parallel coordinates plot
# ─────────────────────────────────────────────────────────────────────────────
def fig_parallel_coordinates(df: pd.DataFrame, out: Path):
    """
    Parallel coordinates showing how parameters relate to outcomes.
    One subplot per improvement metric.
    """
    print("[2] Parallel coordinates plot")
    
    # Aggregate across runs and subjects
    group_cols = ["solver", "noise_type", "nnz", "alpha_SNR"]
    agg_dict = {"emd": "mean"}
    for metric, _ in IMPROVEMENT_METRICS:
        if metric in df.columns:
            agg_dict[metric] = "mean"
    
    agg_df = df.groupby(group_cols).agg(agg_dict).reset_index()
    
    # Create subplot grid: 1 row, 5 columns
    fig, axes = plt.subplots(1, 5, figsize=(30, 6))
    
    for idx, (metric, metric_label) in enumerate(IMPROVEMENT_METRICS):
        ax = axes[idx]
        
        if metric not in agg_df.columns:
            ax.text(0.5, 0.5, f"{metric} not found", ha="center", va="center")
            ax.set_title(metric_label, fontsize=10, fontweight="bold")
            continue
        
        # Define parameter columns
        param_info = [
            ("solver", "Solver", "category"),
            ("noise_type", "Noise", "category"),
            ("nnz", "NNZ", "numeric"),
            ("alpha_SNR", "SNR", "numeric"),
            ("emd", "EMD", "numeric"),
            (metric, "Improve", "numeric")
        ]
        
        x_positions = np.arange(len(param_info))
    
        # Build value-to-position mappings
        value_maps = []
        for col, label, ptype in param_info:
            unique_vals = sorted(agg_df[col].unique())
            if ptype == "category":
                positions = np.linspace(0, 1, len(unique_vals))
                val_map = {val: pos for val, pos in zip(unique_vals, positions)}
            else:
                min_val, max_val = agg_df[col].min(), agg_df[col].max()
                if max_val > min_val:
                    val_map = {val: (val - min_val) / (max_val - min_val) for val in unique_vals}
                else:
                    val_map = {val: 0.5 for val in unique_vals}
            value_maps.append((col, val_map, unique_vals, ptype))
        
        # Map each row to positions
        positions_list = []
        for _, row in agg_df.iterrows():
            positions = [value_maps[i][1][row[value_maps[i][0]]] for i in range(len(param_info))]
            positions_list.append(positions)
        
        # Color by improvement quartile
        quartiles = agg_df[metric].quantile([0.25, 0.5, 0.75])
        colors = []
        for val in agg_df[metric]:
            if val < quartiles[0.25]:
                colors.append("#d7191c")
            elif val < quartiles[0.5]:
                colors.append("#fdae61")
            elif val < quartiles[0.75]:
                colors.append("#a6d96a")
            else:
                colors.append("#1a9641")
    
        # Plot lines
        for positions, color in zip(positions_list, colors):
            ax.plot(x_positions, positions, alpha=0.6, linewidth=1.5, color=color, zorder=2)
        
        # Draw vertical axes (simpler version for space)
        for i, (col, val_map, unique_vals, ptype) in enumerate(value_maps):
            x = x_positions[i]
            ax.plot([x, x], [0, 1], 'k-', linewidth=1.2, alpha=0.3, zorder=1)
            
            # Only show ticks for first and last axis to save space
            if i in [0, len(param_info) - 1]:
                for val in unique_vals:
                    y_pos = val_map[val]
                    ax.plot([x - 0.015, x + 0.015], [y_pos, y_pos], 'k-', linewidth=1, zorder=1)
        
        # Axis labels
        ax.set_xticks(x_positions)
        ax.set_xticklabels([label for _, label, _ in param_info], fontsize=8, fontweight="bold")
        ax.set_xlim(-0.2, len(param_info) - 0.8)
        ax.set_ylim(-0.05, 1.05)
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
        
        ax.set_title(metric_label, fontsize=10, fontweight="bold", pad=10)
    
    fig.suptitle(
        "Parallel Coordinates: Parameter → Performance Relationships (All Metrics)\n"
        f"Each line = mean across runs & subjects (n={len(agg_df)} per metric)",
        fontsize=14, fontweight="bold"
    )
    
    # Reserve right side for legend, then add legend
    fig.tight_layout(rect=[0, 0, 0.88, 1])
    
    # Add single legend for all subplots in the reserved space
    legend_patches = [
        mpatches.Patch(color="#d7191c", label="Q1 (worst)"),
        mpatches.Patch(color="#fdae61", label="Q2"),
        mpatches.Patch(color="#a6d96a", label="Q3"),
        mpatches.Patch(color="#1a9641", label="Q4 (best)")
    ]
    fig.legend(handles=legend_patches, loc="center left", fontsize=9,
              title="Improvement\nQuartile", bbox_to_anchor=(0.89, 0.5))
    
    save_fig(fig, out / "2_parallel_coordinates.png")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 3: PCA dimensionality reduction
# ─────────────────────────────────────────────────────────────────────────────
def fig_pca_clustering(df: pd.DataFrame, out: Path):
    """
    PCA on all calibration + performance metrics.
    One subplot per improvement metric (used for point sizing).
    """
    print("[3] PCA dimensionality reduction")
    
    # Select metrics for PCA: emd + uncertainty + all calibration metrics
    metric_cols = [
        "emd",
        "mean_posterior_std",
        "pre_cal_mean_calibration_error", "post_cal_mean_calibration_error",
        "pre_cal_mean_signed_deviation", "post_cal_mean_signed_deviation",
        "pre_cal_mean_absolute_deviation", "post_cal_mean_absolute_deviation",
        "pre_cal_max_underconfidence_deviation", "post_cal_max_underconfidence_deviation",
        "pre_cal_max_overconfidence_deviation", "post_cal_max_overconfidence_deviation",
    ]
    available = [c for c in metric_cols if c in df.columns]
    
    plot_df = df[available + ["solver", "noise_type"] + 
                 [m for m, _ in IMPROVEMENT_METRICS if m in df.columns]].dropna()
    
    # Standardize and apply PCA once
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(plot_df[available])
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    
    plot_df["PC1"] = X_pca[:, 0]
    plot_df["PC2"] = X_pca[:, 1]
    var_explained = pca.explained_variance_ratio_
    
    # Create subplot grid
    fig, axes = plt.subplots(1, 5, figsize=(25, 5))
    
    solvers = sorted(plot_df["solver"].unique())
    noise_types = sorted(plot_df["noise_type"].unique())
    solver_colors = {s: SOLVER_PALETTE[i % len(SOLVER_PALETTE)] for i, s in enumerate(solvers)}
    noise_markers = {"baseline": "o", "oracle": "s", "spatial_cv": "^"}
    
    for idx, (metric, metric_label) in enumerate(IMPROVEMENT_METRICS):
        ax = axes[idx]
        
        if metric not in plot_df.columns:
            ax.text(0.5, 0.5, f"{metric} not found", ha="center", va="center")
            ax.set_title(metric_label, fontsize=10, fontweight="bold")
            continue
        
        # Plot each solver-noise combination
        for solver in solvers:
            for nt in noise_types:
                mask = (plot_df["solver"] == solver) & (plot_df["noise_type"] == nt)
                subset = plot_df[mask]
                if subset.empty:
                    continue
                
                # Size by improvement
                sizes = subset[metric]
                # Normalize sizes appropriately
                if metric == "improvement_max_overconfidence_deviation":
                    sizes = np.clip(sizes, -100, 1000) * 0.1
                else:
                    sizes = sizes * 2
                sizes = np.clip(sizes, 20, 250)
                
                ax.scatter(
                    subset["PC1"], subset["PC2"],
                    c=solver_colors[solver],
                    marker=noise_markers.get(nt, "o"),
                    s=sizes,
                    alpha=0.6,
                    edgecolors="white",
                    linewidths=0.5
                )
        
        ax.set_xlabel(f"PC1 ({var_explained[0]:.1%})", fontsize=9)
        if idx == 0:
            ax.set_ylabel(f"PC2 ({var_explained[1]:.1%})", fontsize=9)
        ax.grid(alpha=0.3)
        ax.set_title(metric_label, fontsize=10, fontweight="bold", pad=10)
    
    fig.suptitle(
        "PCA: Natural Clustering in Metric Space (All Improvement Metrics)\n"
        f"Based on {len(available)} metrics | Point size ∝ improvement",
        fontsize=14, fontweight="bold"
    )
    
    # Reserve right side for legend, then add legend
    fig.tight_layout(rect=[0, 0, 0.86, 1])
    
    # Legend in the reserved space
    legend_handles = []
    for solver in solvers:
        for nt in noise_types:
            legend_handles.append(
                plt.Line2D([0], [0], marker=noise_markers.get(nt, "o"), color='w',
                          markerfacecolor=solver_colors[solver], markersize=8,
                          label=f"{solver}|{nt}", markeredgecolor='white', markeredgewidth=0.5)
            )
    fig.legend(handles=legend_handles, loc="center left", fontsize=7, ncol=1,
              bbox_to_anchor=(0.87, 0.5), title="Solver | Noise\n(size ∝ improvement)")
    
    save_fig(fig, out / "3_pca_clustering.png")
    
    # Return PCA object and metrics for the loadings plot
    return pca, available, scaler


# ─────────────────────────────────────────────────────────────────────────────
# Figure 3b: PCA feature loadings
# ─────────────────────────────────────────────────────────────────────────────
def fig_pca_loadings(pca, metric_names, out: Path):
    """
    Show which original metrics contribute most to PC1 and PC2.
    Visualizes PCA loadings as bar charts showing feature contributions.
    """
    print("[3b] PCA feature loadings (component contributions)")
    
    if pca is None:
        print("    (skipped: PCA not available)")
        return
    
    # Get loadings (components)
    loadings = pca.components_  # Shape: (2, n_features)
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    
    for idx, (ax, component_name) in enumerate(zip(axes, ["PC1", "PC2"])):
        loading_values = loadings[idx, :]
        var_explained = pca.explained_variance_ratio_[idx]
        
        # Sort by absolute contribution
        sorted_indices = np.argsort(np.abs(loading_values))[::-1]
        sorted_names = [metric_names[i] for i in sorted_indices]
        sorted_values = loading_values[sorted_indices]
        
        # Color by sign (positive vs negative contribution)
        colors = ['#2166ac' if v > 0 else '#b2182b' for v in sorted_values]
        
        # Horizontal bar chart
        y_pos = np.arange(len(sorted_names))
        ax.barh(y_pos, sorted_values, color=colors, alpha=0.8, edgecolor='black', linewidth=0.5)
        
        ax.set_yticks(y_pos)
        ax.set_yticklabels(sorted_names, fontsize=9)
        ax.set_xlabel("Loading (Contribution)", fontsize=11, fontweight="bold")
        ax.set_title(f"{component_name} ({var_explained:.1%} variance explained)", 
                    fontsize=12, fontweight="bold", pad=10)
        ax.axvline(0, color='black', linewidth=1.5, linestyle='-', alpha=0.3)
        ax.grid(axis='x', alpha=0.3)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        # Add value labels inside bars
        for i, (name, val) in enumerate(zip(sorted_names, sorted_values)):
            # Position text inside the bar (at 90% of bar length from origin)
            label_x = val * 0.5 
            ax.text(label_x, i, f'{val:.3f}', va='center', ha='center', 
                   fontsize=8, fontweight='bold', color='black')
    
    # Add legend
    legend_elements = [
        mpatches.Patch(color='#2166ac', label='Positive contribution'),
        mpatches.Patch(color='#b2182b', label='Negative contribution')
    ]
    fig.legend(handles=legend_elements, loc ='upper center', ncol=2, 
              fontsize=10, bbox_to_anchor=(0.5, 0.97))
    
    fig.suptitle(
        "PCA Feature Loadings: Which Metrics Drive Each Component?\n"
        "Higher absolute values = stronger influence on component",
        fontsize=14, fontweight="bold", y=1.02
    )
    
    fig.tight_layout()
    save_fig(fig, out / "3b_pca_feature_loadings.png")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 4: Parameter interaction effects
# ─────────────────────────────────────────────────────────────────────────────
def fig_interaction_effects(df: pd.DataFrame, out: Path):
    """
    Parameter interaction effects across all improvement metrics.
    Grid: 4 parameter pairs (rows) × 5 metrics (columns).
    """
    print("[4] Parameter interaction effects")
    
    # Define parameter pairs
    param_pairs = [
        ("alpha_SNR", "nnz", "SNR", "NNZ"),
        ("alpha_SNR", "noise_type", "SNR", "Noise"),
        ("nnz", "noise_type", "NNZ", "Noise"),
        ("solver", "noise_type", "Solver", "Noise"),
    ]
    
    # Create grid: 4 rows × 5 columns
    fig, axes = plt.subplots(4, 5, figsize=(25, 16))
    
    for col_idx, (metric, metric_label) in enumerate(IMPROVEMENT_METRICS):
        if metric not in df.columns:
            for row_idx in range(len(param_pairs)):
                axes[row_idx, col_idx].text(0.5, 0.5, "N/A", ha="center", va="center")
            continue
        
        for row_idx, (param1, param2, label1, label2) in enumerate(param_pairs):
            ax = axes[row_idx, col_idx]
            
            # Aggregate
            grouped = df.groupby([param1, param2])[metric].mean().reset_index()
            pivot = grouped.pivot(index=param2, columns=param1, values=metric)
            
            # Adaptive colorbar limits
            vmin, vmax = pivot.values.min(), pivot.values.max()
            if metric == "improvement_max_overconfidence_deviation":
                vmin, vmax = max(vmin, -1000), min(vmax, 5000)
            
            # Heatmap
            im = ax.imshow(pivot.values, cmap=DIVERGING_CMAP, aspect="auto",
                          vmin=vmin, vmax=vmax)
            
            ax.set_xticks(np.arange(len(pivot.columns)))
            ax.set_yticks(np.arange(len(pivot.index)))
            ax.set_xticklabels(pivot.columns, fontsize=7)
            ax.set_yticklabels(pivot.index, fontsize=7)
            
            # Annotate cells
            for i in range(len(pivot.index)):
                for j in range(len(pivot.columns)):
                    val = pivot.values[i, j]
                    if not np.isnan(val):
                        mid = (vmin + vmax) / 2
                        color = "white" if abs(val - mid) > (vmax - vmin) * 0.3 else "black"
                        text = f"{val:.0f}" if abs(val) > 100 else f"{val:.1f}"
                        ax.text(j, i, text, ha="center", va="center",
                               color=color, fontsize=6, fontweight="bold")
            
            # Labels
            if row_idx == len(param_pairs) - 1:
                ax.set_xlabel(label1, fontsize=8, fontweight="bold")
            if col_idx == 0:
                ax.set_ylabel(label2, fontsize=8, fontweight="bold")
            
            # Title only on top row
            if row_idx == 0:
                ax.set_title(metric_label, fontsize=9, fontweight="bold", pad=8)
            
            # Row label on left column
            if col_idx == 0:
                ax.text(-0.5, 0.5, f"{label1}×{label2}", 
                       transform=ax.transAxes, rotation=90,
                       va="center", ha="right", fontsize=9, fontweight="bold")
    
    # Add shared colorbar on the right
    fig.subplots_adjust(right=0.92)
    cbar_ax = fig.add_axes([0.94, 0.15, 0.02, 0.7])
    fig.colorbar(im, cax=cbar_ax, label="Mean Improvement (%)")
    
    fig.suptitle(
        "Parameter Interaction Effects: All Improvement Metrics\n"
        "Rows = parameter pairs | Columns = improvement metrics",
        fontsize=14, fontweight="bold", y=0.995
    )
    
    fig.tight_layout(rect=[0, 0, 0.92, 0.98])
    save_fig(fig, out / "4_interaction_effects.png")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 5: Ridge plots showing distribution shifts
# ─────────────────────────────────────────────────────────────────────────────
def fig_ridge_plots(df: pd.DataFrame, out: Path):
    """
    Ridge plots showing distribution shifts.
    Grid: 4 parameters (rows) × 5 metrics (columns).
    """
    print("[5] Ridge plots: distribution shifts across parameters")
    
    from scipy.stats import gaussian_kde
    
    # Parameters to visualize
    params = [
        ("solver", "Solver"),
        ("alpha_SNR", "SNR"),
        ("nnz", "NNZ"),
        ("noise_type", "Noise"),
    ]
    
    # Create grid: 4 rows × 5 columns
    fig, axes = plt.subplots(4, 5, figsize=(25, 16))
    
    # Track colors and groups for each row to create legends later
    row_legend_data = []
    
    for col_idx, (metric, metric_label) in enumerate(IMPROVEMENT_METRICS):
        if metric not in df.columns:
            for row_idx in range(len(params)):
                axes[row_idx, col_idx].text(0.5, 0.5, "N/A", ha="center", va="center")
            continue
        
        for row_idx, (param, param_label) in enumerate(params):
            ax = axes[row_idx, col_idx]
            
            groups = sorted(df[param].unique())
            colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(groups)))
            
            # Store legend info for first column only
            if col_idx == 0:
                row_legend_data.append((groups, colors, param_label))
            
            y_offset = 0
            y_spacing = 0.5
            
            for i, group_val in enumerate(groups):
                subset = df[df[param] == group_val][metric].dropna()
                if len(subset) < 2:
                    continue
                
                try:
                    kde = gaussian_kde(subset, bw_method=0.3)
                    x_range = np.linspace(subset.min() - 5, subset.max() + 5, 200)
                    density = kde(x_range)
                    density = density / density.max() * y_spacing * 0.8
                    
                    ax.fill_between(x_range, y_offset, y_offset + density,
                                   color=colors[i], alpha=0.7)
                    ax.plot(x_range, y_offset + density, color=colors[i],
                           linewidth=1.5, alpha=0.9)
                    
                    # Mean line
                    mean_val = subset.mean()
                    ax.plot([mean_val, mean_val], [y_offset, y_offset + density.max()],
                           'k--', linewidth=1, alpha=0.5)
                    
                    y_offset += y_spacing
                except:
                    pass
            
            # Formatting
            if row_idx == len(params) - 1:
                ax.set_xlabel("Improvement (%)", fontsize=8)
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_edgecolor("#333333")
                spine.set_linewidth(1)
            ax.grid(axis="x", alpha=0.2)
            
            # Title only on top row
            if row_idx == 0:
                ax.set_title(metric_label, fontsize=9, fontweight="bold", pad=8)
            
            # Row label on leftmost column
            if col_idx == 0:
                ax.set_ylabel(param_label, fontsize=8, fontweight="bold")
    
    fig.suptitle(
        "Ridge Plots: Distribution Shifts Across All Metrics\n"
        "Rows = parameters | Columns = improvement metrics | Dashed lines = means",
        fontsize=14, fontweight="bold", y=0.995
    )
    
    # Reserve right side for legends
    fig.tight_layout(rect=[0, 0, 0.92, 0.98])
    
    # Add one legend per row on the right side
    for row_idx, (groups, colors, param_label) in enumerate(row_legend_data):
        # Calculate vertical position for this row's legend (centered on row)
        row_center = 1 - (row_idx + 0.5) / len(params)
        
        legend_handles = [mpatches.Patch(color=colors[i], label=str(g)) 
                         for i, g in enumerate(groups)]
        fig.legend(handles=legend_handles, loc="center left", fontsize=7,
                  bbox_to_anchor=(0.925, row_center), 
                  title=param_label, frameon=True, framealpha=0.9)
    
    save_fig(fig, out / "5_ridge_plots.png")


# ─────────────────────────────────────────────────────────────────────────────
def main():
    print("  CaliBrain – Cluster Exploration Figures")
    print("=" * 65)
    
    df = pd.read_csv(CSV_PATH)
    print(f"Loaded {len(df)} rows from CSV\n")
    
    out = Path(OUTPUT_DIR) if OUTPUT_DIR else (
        Path(CSV_PATH).parent / "figures" / "cluster_exploration"
    )
    out.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {out}\n")
    
    # Generate figures
    fig_improvement_heatmap(df, out)
    fig_parallel_coordinates(df, out)
    pca_obj, pca_metrics, scaler = fig_pca_clustering(df, out)
    fig_pca_loadings(pca_obj, pca_metrics, out)
    fig_interaction_effects(df, out)
    fig_ridge_plots(df, out)
    
    print("\n" + "=" * 65)
    print("All cluster exploration figures complete.")


if __name__ == "__main__":
    main()
