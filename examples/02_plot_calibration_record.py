"""
02. Plot a calibration JSON record
==================================

This example writes a compact calibration JSON record, loads it with
CaliBrain's storage helper, and plots pre- and post-calibration curves. The same
record structure is produced by the calibration workflow.
"""

# sphinx_gallery_thumbnail_number = 1

# %%
# Create a compact calibration record
# -----------------------------------

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from calibrain.calibration_storage import load_calibration_record, save_calibration_record


OUTPUT_DIR = Path("results/examples/02_plot_calibration_record")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

nominal = np.linspace(0.0, 1.0, 11)
pre_empirical = np.array([0.0, 0.03, 0.10, 0.19, 0.31, 0.43, 0.55, 0.66, 0.77, 0.88, 1.0])
post_empirical = np.array([0.0, 0.09, 0.20, 0.31, 0.40, 0.51, 0.61, 0.70, 0.81, 0.91, 1.0])

record_path = save_calibration_record(
    output_dir=OUTPUT_DIR,
    record_name="calibration_run-demo",
    metadata={
        "solver": "gamma_map",
        "noise_type": "oracle",
        "calibration_mode": "post_oracle",
        "orientation_type": "fixed",
    },
    pre_calibration={
        "nominal_coverages": nominal,
        "empirical_coverages": pre_empirical,
        "n_observations": 240,
    },
    post_calibration={
        "nominal_coverages": nominal,
        "empirical_coverages": post_empirical,
        "recalibrated_nominal_coverages": nominal,
    },
)

# %%
# Load and plot the record
# ------------------------

record = load_calibration_record(record_path)
pre = record["pre_calibration"]
post = record["post_calibration"]

fig, ax = plt.subplots(figsize=(5, 5))
ax.plot([0, 1], [0, 1], "--", color="0.5", label="perfect calibration")
ax.plot(pre["nominal_coverages"], pre["empirical_coverages"], "o-", label="precal")
ax.plot(post["nominal_coverages"], post["empirical_coverages"], "o-", label="post_oracle")
ax.set(
    xlabel="Nominal coverage",
    ylabel="Empirical coverage",
    title="Calibration record curves",
    xlim=(0, 1),
    ylim=(0, 1),
)
ax.set_aspect("equal", adjustable="box")
ax.grid(True, linestyle="--", alpha=0.4)
ax.legend()
fig.tight_layout()
fig.savefig(OUTPUT_DIR / "calibration_record_curves.png", dpi=150)

print(record["metadata"])
