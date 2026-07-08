from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from calibrain import (
    BMN,
    LeadfieldBuilder,
    MetricEvaluator,
    SensorSimulator,
    SourceEstimator,
    SourceSimulator,
    UncertaintyCalibrator,
    UncertaintyEstimator,
)


def main() -> None:
    plt.style.use("default")
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 11,
            "axes.labelsize": 11,
            "axes.titlesize": 11,
            "legend.fontsize": 10,
            "axes.linewidth": 1.0,
            "xtick.direction": "out",
            "ytick.direction": "out",
            "xtick.major.size": 4,
            "ytick.major.size": 4,
            "xtick.major.width": 1.0,
            "ytick.major.width": 1.0,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "grid.color": "0.85",
            "grid.linestyle": "-",
            "grid.linewidth": 0.8,
        }
    )

    output_path = Path("docs/source/_static/readme_example.png")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    source_simulator = SourceSimulator()
    leadfield_builder = LeadfieldBuilder(leadfield_dir="unused")
    sensor_simulator = SensorSimulator()
    nominal_coverages = np.linspace(0.0, 1.0, 11)
    uncertainty = UncertaintyEstimator(nominal_coverages=nominal_coverages)
    metric_evaluator = MetricEvaluator(uncertainty, nominal_coverages=nominal_coverages)
    calibrator = UncertaintyCalibrator(uncertainty, metric_evaluator)

    x_true, _ = source_simulator.simulate(n_sources=64, nnz=4, seed=0)
    leadfield = leadfield_builder.get_leadfield(
        retrieve_mode="random",
        orientation_type="fixed",
        n_sensors=20,
        n_sources=x_true.shape[0],
    )
    _, y_noisy, noise, _ = sensor_simulator.simulate(x_true, leadfield, seed=0)
    result = SourceEstimator(solver=BMN, noise_var=float(np.var(noise))).fit(leadfield, y_noisy).predict()

    posterior_var = uncertainty.posterior_variance_from_cov(result["posterior_cov"])
    pre_curve = uncertainty.calibration_curve_intervals_aggregated(
        x_true=x_true,
        x_hat=result["posterior_mean"],
        posterior_var=posterior_var,
    )
    data = {
        "x_true": x_true,
        "x_hat": result["posterior_mean"],
        "posterior_var": posterior_var,
    }
    mapping = calibrator.fit_mapping(train_data=data)
    post_results = calibrator.evaluate_with_mapping(test_data=data)

    fig, ax = plt.subplots(figsize=(5.3, 4.3), constrained_layout=True)
    ax.plot([0, 1], [0, 1], linestyle="--", color="0.35", linewidth=1.4, label="perfect calibration")
    ax.plot(
        pre_curve["nominal_coverages"],
        pre_curve["empirical_coverages"],
        marker="o",
        markersize=5.5,
        linewidth=1.8,
        color="#1f77b4",
        label="before calibration",
    )
    ax.plot(
        post_results["post_calibration"]["nominal_coverages"],
        post_results["post_calibration"]["empirical_coverages"],
        marker="o",
        markersize=5.5,
        linewidth=1.8,
        color="#d62728",
        label="after calibration",
    )
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.02)
    ax.set_xlabel("Nominal coverage")
    ax.set_ylabel("Empirical coverage")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, which="major")
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), frameon=False)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Saved figure to {output_path.resolve()}")


if __name__ == "__main__":
    main()
