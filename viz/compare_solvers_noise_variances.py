from pathlib import Path
from typing import List, Dict, Optional
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

def plot_summary_metrics_by_snr_solver_fixed(df: pd.DataFrame, output_path: Path, nnz: int, orientation_type: str, solver: str) -> None:
    # Filter dataframe
    df = df[(df['nnz'] == nnz) & (df['orientation_type'] == orientation_type) & (df['solver'] == solver)]
    metrics = ALL_METRICS
    noise_types = sorted(df['noise_type'].dropna().unique())
    snr_values = np.sort(df['alpha_SNR'].dropna().unique())
    n_metrics = len(metrics)
    nrows = 2
    ncols = int(np.ceil(n_metrics / nrows))
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(6 * ncols, 4 * nrows), sharex=True)
    axes = np.atleast_1d(axes).ravel()
    # Use a more distinguishable color palette (Set2, tab20)
    color_map = plt.get_cmap('Set2') if len(noise_types) <= 8 else plt.get_cmap('tab20')
    noise_colors = {nt: color_map(i % color_map.N) for i, nt in enumerate(noise_types)}
    line_styles = {'pre': '-', 'post': '--'}
    for idx, metric in enumerate(metrics):
        ax = axes[idx]
        for i, noise_type in enumerate(noise_types):
            df_noise = df[df['noise_type'] == noise_type]
            for prepost in (['pre', 'post'] if metric in CALIBRATION_METRICS else ['pre']):
                if metric in CALIBRATION_METRICS:
                    col = f'{prepost}_cal_{metric}'
                else:
                    col = metric
                means = []
                for snr in snr_values:
                    vals = df_noise[df_noise['alpha_SNR'] == snr][col].dropna()
                    means.append(vals.mean() if not vals.empty else np.nan)
                label = f"{noise_type} {prepost}" if metric in CALIBRATION_METRICS else noise_type
                style = line_styles[prepost] if metric in CALIBRATION_METRICS else '-'
                ax.plot(snr_values, means, style, color=noise_colors[noise_type], label=label)
        ax.set_title(METRIC_LABELS.get(metric, metric))
        ax.set_xlabel('alpha_SNR')
        ax.set_ylabel(METRIC_LABELS.get(metric, metric))
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=10)
    for idx in range(len(metrics), len(axes)):
        axes[idx].axis('off')
    fig.suptitle(f"Summary of Calibration and Evaluation Metrics by SNR\n(nnz={nnz}, orientation_type={orientation_type}, solver={solver})", fontsize=18, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(output_path, dpi=300)
    print(f"Saved summary figure: {output_path}")
    plt.close(fig)


def summarize_metric(df: pd.DataFrame, metric: str, group_col: str) -> pd.DataFrame:
    #!/usr/bin/env python
    """
    Compare all inverse solvers and noise variance methods across different setups.
    Generates two figures:
    1. For each solver, compares metrics across noise variance methods.
    2. For each noise variance method, compares metrics across solvers.

    Usage: Paste your CSV file paths into the CSV_FILES list in main().
    """

    # --- Configuration ---
    METRIC_LABELS = {
        "mean_signed_deviation": "Mean Signed Deviation",
        "mean_absolute_deviation": "Mean Absolute Deviation",
        "max_underconfidence_deviation": "Max Underconfidence Deviation",
        "max_overconfidence_deviation": "Max Overconfidence Deviation",
        "mean_posterior_std": "Mean Posterior Std",
        "emd": "Earth Mover's Distance",
        "gamma": "Gamma Norm",
    }
    CALIBRATION_METRICS = [
        "mean_signed_deviation",
        "mean_absolute_deviation",
        "max_underconfidence_deviation",
        "max_overconfidence_deviation",
    ]
    EVALUATION_METRICS = [
        "mean_posterior_std",
        "emd",
        "gamma",
    ]
    ALL_METRICS = CALIBRATION_METRICS + EVALUATION_METRICS
    PRE_COLOR = "#1f77b4"
    POST_COLOR = "#d62728"

    # --- Utility Functions ---
    def load_csv_with_context(csv_path: Path) -> pd.DataFrame:
        df = pd.read_csv(csv_path)
        df["_csv_path"] = str(csv_path)
        return df

    def collect_experiment_data(csv_files: List[str]) -> pd.DataFrame:
        dfs = [load_csv_with_context(Path(f)) for f in csv_files]
        df = pd.concat(dfs, ignore_index=True)
        # Clean inf/-inf
        df = df.replace([np.inf, -np.inf], np.nan)
        return df

    # --- Summary Functions ---
    def summarize_metric(df: pd.DataFrame, metric: str, group_col: str) -> pd.DataFrame:
        grouped = df.groupby(group_col)
        rows = []
        for group_value, group in grouped:
            if metric in CALIBRATION_METRICS:
                pre_mean = group[f"pre_cal_{metric}"].mean()
                pre_std = group[f"pre_cal_{metric}"].std(ddof=0)
                post_mean = group[f"post_cal_{metric}"].mean()
                post_std = group[f"post_cal_{metric}"].std(ddof=0)
                rows.append({
                    group_col: group_value,
                    "pre_mean": pre_mean,
                    "pre_std": pre_std,
                    "post_mean": post_mean,
                    "post_std": post_std,
                })
            else:
                value_mean = group[metric].mean()
                value_std = group[metric].std(ddof=0)
                rows.append({
                    group_col: group_value,
                    "value_mean": value_mean,
                    "value_std": value_std,
                })
        return pd.DataFrame(rows)
# --- Summary Plotting Functions ---
#!/usr/bin/env python
"""
Compare all inverse solvers and noise variance methods across different setups.
Generates two figures:
1. For each solver, compares metrics across noise variance methods.
2. For each noise variance method, compares metrics across solvers.

Usage: Paste your CSV file paths into the CSV_FILES list in main().
"""
from pathlib import Path
from typing import List, Dict, Optional
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

# --- Configuration ---
METRIC_LABELS = {
    "mean_signed_deviation": "Mean Signed Deviation",
    "mean_absolute_deviation": "Mean Absolute Deviation",
    "max_underconfidence_deviation": "Max Underconfidence Deviation",
    "max_overconfidence_deviation": "Max Overconfidence Deviation",
    "mean_posterior_std": "Mean Posterior Std",
    "emd": "Earth Mover's Distance",
    "gamma": "Gamma Norm",
}
CALIBRATION_METRICS = [
    "mean_signed_deviation",
    "mean_absolute_deviation",
    "max_underconfidence_deviation",
    "max_overconfidence_deviation",
]
EVALUATION_METRICS = [
    "mean_posterior_std",
    "emd",
    "gamma",
]
ALL_METRICS = CALIBRATION_METRICS + EVALUATION_METRICS
PRE_COLOR = "#1f77b4"
POST_COLOR = "#d62728"

# --- Utility Functions ---
def load_csv_with_context(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df["_csv_path"] = str(csv_path)
    return df

def collect_experiment_data(csv_files: List[str]) -> pd.DataFrame:
    dfs = [load_csv_with_context(Path(f)) for f in csv_files]
    df = pd.concat(dfs, ignore_index=True)
    # Clean inf/-inf
    df = df.replace([np.inf, -np.inf], np.nan)
    return df

def plot_by_solver(df: pd.DataFrame, output_path: Path) -> None:
    solvers = sorted(df["solver"].dropna().unique())
    noise_types = sorted(df["noise_type"].dropna().unique())
    metrics = ALL_METRICS
    snr_values = np.sort(df['alpha_SNR'].dropna().unique())
    nrows = len(solvers)
    ncols = len(metrics)
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(5 * ncols, 4 * nrows), sharex=True)
    axes = np.atleast_1d(axes).reshape(nrows, ncols)
    color_map = plt.get_cmap('Set2') if len(noise_types) <= 8 else plt.get_cmap('tab20')
    noise_colors = {nt: color_map(i % color_map.N) for i, nt in enumerate(noise_types)}
    for i, solver in enumerate(solvers):
        df_solver = df[df["solver"] == solver]
        for idx, metric in enumerate(metrics):
            ax = axes[i, idx]
            bar_width = 0.35
            x = np.arange(len(snr_values))
            for k, noise_type in enumerate(noise_types):
                df_noise = df_solver[df_solver['noise_type'] == noise_type]
                for j, prepost in enumerate(['pre', 'post'] if metric in CALIBRATION_METRICS else ['pre']):
                    if metric in CALIBRATION_METRICS:
                        col = f'{prepost}_cal_{metric}'
                    else:
                        col = metric
                    means = []
                    stds = []
                    for snr in snr_values:
                        vals = df_noise[df_noise['alpha_SNR'] == snr][col].dropna()
                        means.append(vals.mean() if not vals.empty else np.nan)
                        stds.append(vals.std(ddof=0) if not vals.empty else 0)
                    label = f"{noise_type} {prepost}" if metric in CALIBRATION_METRICS else noise_type
                    color = noise_colors[noise_type]
                    ax.bar(x + (k + j * 0.5) * bar_width, means, bar_width, yerr=stds, label=label, color=color, alpha=0.8, capsize=4)
            ax.set_xticks(x + bar_width)
            ax.set_xticklabels([str(snr) for snr in snr_values])
            ax.set_title(METRIC_LABELS.get(metric, metric))
            ax.set_xlabel('alpha_SNR')
            ax.set_ylabel(METRIC_LABELS.get(metric, metric))
            ax.grid(True, alpha=0.25)
            ax.legend(fontsize=10)
    fig.suptitle("Metrics by Solver (across Noise Variance Methods)", fontsize=18, fontweight="bold")
    # Show fixed parameters as subtitle
    nnz_val = df['nnz'].iloc[0] if 'nnz' in df.columns and not df['nnz'].isnull().all() else 'N/A'
    orientation_val = df['orientation_type'].iloc[0] if 'orientation_type' in df.columns and not df['orientation_type'].isnull().all() else 'N/A'
    fig.text(0.5, 0.94, f"(nnz={nnz_val}, orientation_type={orientation_val})", ha='center', fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(output_path, dpi=300)
    print(f"Saved figure: {output_path}")
    plt.close(fig)

def plot_by_noise_type(df: pd.DataFrame, output_path: Path) -> None:
    solvers = sorted(df["solver"].dropna().unique())
    noise_types = sorted(df["noise_type"].dropna().unique())
    metrics = ALL_METRICS
    nrows = len(noise_types)
    ncols = len(metrics)
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(5 * ncols, 4 * nrows), sharex=True)
    axes = np.atleast_1d(axes).reshape(nrows, ncols)
    for i, noise_type in enumerate(noise_types):
        df_noise = df[df["noise_type"] == noise_type]
        for j, metric in enumerate(metrics):
            ax = axes[i, j]
            for k, solver in enumerate(solvers):
                df_sub = df_noise[df_noise["solver"] == solver]
                if metric in CALIBRATION_METRICS:
                    pre_vals = df_sub[f"pre_cal_{metric}"].dropna()
                    post_vals = df_sub[f"post_cal_{metric}"].dropna()
                    if len(pre_vals) > 0:
                        bp_pre = ax.boxplot(pre_vals, positions=[k - 0.15], widths=0.15, patch_artist=True, label="Pre-calibration")
                        for patch in bp_pre['boxes']:
                            patch.set_facecolor(PRE_COLOR)
                    if len(post_vals) > 0:
                        bp_post = ax.boxplot(post_vals, positions=[k + 0.15], widths=0.15, patch_artist=True, label="Post-calibration")
                        for patch in bp_post['boxes']:
                            patch.set_facecolor(POST_COLOR)
                else:
                    vals = df_sub[metric].dropna()
                    if len(vals) > 0:
                        bp = ax.boxplot(vals, positions=[k], widths=0.18, patch_artist=True, label=solver)
                        for patch in bp['boxes']:
                            patch.set_facecolor(PRE_COLOR)
            # Add legend for calibration metrics
            if metric in CALIBRATION_METRICS:
                handles = [
                    Patch(facecolor=PRE_COLOR, label="Pre-calibration"),
                    Patch(facecolor=POST_COLOR, label="Post-calibration"),
                ]
                ax.legend(handles=handles, loc="upper right", fontsize=10)
            else:
                ax.legend(handles=[Patch(facecolor=PRE_COLOR, label=solver)], loc="upper right", fontsize=10)
            ax.set_xticks(range(len(solvers)))
            ax.set_xticklabels(solvers)
            ax.set_title(f"{noise_type} - {METRIC_LABELS.get(metric, metric)}")
    fig.suptitle("Metrics by Noise Variance Method (across Solvers)", fontsize=18, fontweight="bold")
    # Show fixed parameters as subtitle
    nnz_val = df['nnz'].iloc[0] if 'nnz' in df.columns and not df['nnz'].isnull().all() else 'N/A'
    orientation_val = df['orientation_type'].iloc[0] if 'orientation_type' in df.columns and not df['orientation_type'].isnull().all() else 'N/A'
    fig.text(0.5, 0.94, f"(nnz={nnz_val}, orientation_type={orientation_val})", ha='center', fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(output_path, dpi=300)
    print(f"Saved figure: {output_path}")
    plt.close(fig)

# --- Main ---
def main():
    # Example usage for new summary figure
    # Paste your CSV file paths here
    CSV_FILES = [
        'results/benchmark_results/benchmark_results_20251128_160842_eloreta_oracle.csv',
        'results/benchmark_results/benchmark_results_20251128_161645_eloreta_baseline.csv',
        'results/benchmark_results/benchmark_results_20251128_162514_eloreta_spatialCV.csv',
        'results/benchmark_results/benchmark_results_20251128_160915_sflex_oracle.csv',
        'results/benchmark_results/benchmark_results_20251128_173356_sflex_baseline.csv',
        'results/benchmark_results/benchmark_results_20251128_185913_sflex_spatialCV.csv',
        'results/benchmark_results/benchmark_results_20251128_160858_bmn_oracle.csv',
        'results/benchmark_results/benchmark_results_20251128_185729_bmn_baseline.csv',
    ]
    comparison_dir = Path("results/benchmark_results/noise_variance_comparisons")
    comparison_dir.mkdir(parents=True, exist_ok=True)
    df = collect_experiment_data(CSV_FILES)

    # Create summary figures fixing solver, combining all noise variance methods
    for solver in sorted(df['solver'].dropna().unique()):
        output_solver_fixed = comparison_dir / f"summary_by_snr_solver={solver}.png"
        plot_summary_metrics_by_snr_solver_fixed(
            df,
            output_solver_fixed,
            nnz=5,
            orientation_type="fixed",
            solver=solver
        )

if __name__ == "__main__":
    main()
