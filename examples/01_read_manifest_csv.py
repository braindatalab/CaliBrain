"""
01. Read a manifest CSV
=======================

This example creates a lightweight manifest CSV, loads it with CaliBrain's
manifest helper, and visualizes how many posterior summaries are available per
solver. In production workflows, the manifest is written by data generation and
used by aggregation to select runs.
"""

# sphinx_gallery_thumbnail_number = 1

# %%
# Create a small manifest
# -----------------------
#
# A CaliBrain manifest records one posterior-summary file per generated run,
# together with metadata used for filtering and grouping.

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from calibrain.run_manifest import load_manifest_csv


OUTPUT_DIR = Path("results/examples/01_read_manifest_csv")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
manifest_path = OUTPUT_DIR / "manifest.csv"

rows = [
    ("posterior/run-001.h5", 1, 11, "gamma_map", "oracle", 0.9, 3),
    ("posterior/run-002.h5", 2, 12, "gamma_map", "oracle", 0.9, 3),
    ("posterior/run-003.h5", 3, 13, "gamma_map_sflex", "oracle", 0.9, 3),
    ("posterior/run-004.h5", 4, 14, "BMN", "oracle", 0.9, 3),
    ("posterior/run-005.h5", 5, 15, "BMN", "oracle", 0.9, 3),
    ("posterior/run-006.h5", 6, 16, "BMN", "oracle", 0.9, 3),
]

with manifest_path.open("w", encoding="utf-8") as handle:
    handle.write(
        "posterior_summary,global_run_id,run_id,solver,noise_type,alpha_SNR,nnz\n"
    )
    for row in rows:
        handle.write(",".join(map(str, row)) + "\n")

# %%
# Load and inspect rows
# ---------------------

manifest_rows = load_manifest_csv(manifest_path)
for row in manifest_rows[:2]:
    print(row.summary_path, row.metadata)

# %%
# Plot solver counts
# ------------------
#
# This plot is intentionally simple: it shows the kind of manifest-level sanity
# check that is useful before running aggregation.

solvers = [row.metadata["solver"] for row in manifest_rows]
unique_solvers, counts = np.unique(solvers, return_counts=True)

fig, ax = plt.subplots(figsize=(5.5, 3.5))
ax.bar(unique_solvers, counts, width=1.0, color="#4C72B0")
ax.set(
    xlabel="Solver",
    ylabel="Number of manifest rows",
    title="Posterior summaries per solver",
)
ax.grid(axis="y", linestyle="--", alpha=0.4)
fig.tight_layout()
fig.savefig(OUTPUT_DIR / "manifest_solver_counts.png", dpi=150)
