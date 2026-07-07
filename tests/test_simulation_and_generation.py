from pathlib import Path

import numpy as np
import pytest
from mne.io.constants import FIFF

from calibrain.data_generation import DataGenerator
from calibrain.sensor_simulation import SensorSimulator
from calibrain.source_simulation import SourceSimulator


def _make_erp_config():
    return {
        "tmin": -0.5,
        "tmax": 0.5,
        "stim_onset": 0.0,
        "sfreq": 250,
        "fmin": 1,
        "fmax": 5,
        "amplitude_distribution": {
            "median": 20.0,
            "sigma": 0.2,
            "clip": (2.5, 50.0),
        },
        "random_erp_timing": False,
        "erp_min_length": 82,
    }


def test_source_simulator_fixed_returns_expected_shape_and_active_count():
    simulator = SourceSimulator(ERP_config=_make_erp_config())

    x, active_sources = simulator.simulate(
        n_sources=8,
        nnz=3,
        orientation_type="fixed",
        seed=7,
    )

    assert x.shape == (8, 250)
    assert len(active_sources) == 3
    assert len(np.unique(active_sources)) == 3
    assert np.count_nonzero(np.linalg.norm(x, axis=1)) == 3


def test_source_simulator_free_eeg_returns_expected_shape():
    simulator = SourceSimulator(ERP_config=_make_erp_config())

    x, active_sources = simulator.simulate(
        n_sources=6,
        nnz=2,
        orientation_type="free",
        coil_type=FIFF.FIFFV_COIL_EEG,
        seed=11,
    )

    assert x.shape == (6, 3, 250)
    assert len(active_sources) == 2
    assert np.count_nonzero(np.linalg.norm(x.reshape(6, -1), axis=1)) == 2


def test_source_simulator_free_meg_returns_expected_shape():
    simulator = SourceSimulator(ERP_config=_make_erp_config())

    x, active_sources = simulator.simulate(
        n_sources=5,
        nnz=2,
        orientation_type="free",
        coil_type=FIFF.FIFFV_COIL_VV_MAG_T1,
        seed=13,
    )

    assert x.shape == (5, 2, 250)
    assert len(active_sources) == 2
    assert np.count_nonzero(np.linalg.norm(x.reshape(5, -1), axis=1)) == 2


def test_sensor_simulator_fixed_projection_and_noise_extremes():
    simulator = SensorSimulator()
    x = np.array([[1.0, 2.0], [3.0, 4.0]])
    L = np.array([[1.0, 0.0], [0.0, 2.0], [1.0, 1.0]])

    y_clean, y_noisy, noise, eta = simulator.simulate(
        x=x,
        L=L,
        alpha_SNR=1.0,
        sensor_white_noise_std=1.0,
        seed=3,
    )

    np.testing.assert_allclose(y_clean, L @ x)
    np.testing.assert_allclose(y_noisy, y_clean)
    np.testing.assert_allclose(noise, np.zeros_like(y_clean))
    assert eta == pytest.approx(0.0)


def test_sensor_simulator_free_projection_matches_einsum():
    simulator = SensorSimulator()
    x = np.arange(2 * 3 * 4, dtype=float).reshape(2, 3, 4)
    L = np.arange(5 * 2 * 3, dtype=float).reshape(5, 2, 3)

    y_clean, _, _, _ = simulator.simulate(
        x=x,
        L=L,
        alpha_SNR=1.0,
        sensor_white_noise_std=1.0,
        seed=5,
    )

    expected = np.einsum("mnk,nkt->mt", L, x)
    np.testing.assert_allclose(y_clean, expected)


def test_data_generator_run_returns_dataframe_with_stubbed_worker(tmp_path, monkeypatch):
    source_simulator = SourceSimulator(ERP_config=_make_erp_config())
    sensor_simulator = SensorSimulator()

    class DummyLeadfieldBuilder:
        pass

    generator = DataGenerator(
        solver=lambda **kwargs: None,
        solver_param_grid={"solver_name": ["dummy_solver"]},
        data_param_grid={"orientation_type": ["fixed"], "nnz": [2]},
        noise_param_grid={"noise_type": ["oracle"]},
        ERP_config=_make_erp_config(),
        source_simulator=source_simulator,
        leadfield_builder=DummyLeadfieldBuilder(),
        sensor_simulator=sensor_simulator,
        posterior_dir=tmp_path / "posterior",
    )

    def fake_execute_single_run(
        run_id,
        nruns,
        total_runs,
        solver_params,
        data_params,
        noise_params,
        seed,
        fig_path,
        global_run_id,
        global_total_runs,
    ):
        return {
            "run_id": run_id,
            "global_run_id": global_run_id,
            "seed": seed,
            "solver": solver_params["solver_name"],
            "orientation_type": data_params["orientation_type"],
            "nnz": data_params["nnz"],
            "noise_type": noise_params["noise_type"],
        }

    monkeypatch.setattr(generator, "_execute_single_run", fake_execute_single_run)

    results = generator.run(nruns=2, fig_path=str(tmp_path / "figures"), n_jobs=1)

    assert list(results["run_id"]) == [1, 2]
    assert list(results["global_run_id"]) == [1, 2]
    assert set(results["solver"]) == {"dummy_solver"}
    assert set(results["orientation_type"]) == {"fixed"}
    assert set(results["nnz"]) == {2}
    assert set(results["noise_type"]) == {"oracle"}


def test_data_generator_create_experiment_directory_respects_order(tmp_path):
    generator = DataGenerator(
        solver=lambda **kwargs: None,
        solver_param_grid={},
        data_param_grid={},
        noise_param_grid={},
        ERP_config=_make_erp_config(),
        source_simulator=SourceSimulator(ERP_config=_make_erp_config()),
        leadfield_builder=object(),
        sensor_simulator=SensorSimulator(),
    )

    experiment_dir = generator._create_experiment_directory(
        base_dir=str(tmp_path),
        params={
            "solver": "BMN",
            "noise_type": "oracle",
            "orientation_type": "fixed",
            "alpha_SNR": 0.9,
            "nnz": 3,
            "seed": 42,
        },
        desired_order=["solver", "noise_type", "orientation_type", "alpha_SNR", "nnz", "seed"],
    )

    experiment_path = Path(experiment_dir)
    assert experiment_path.exists()
    assert experiment_path.parts[-6:] == (
        "solver=BMN",
        "noise_type=oracle",
        "orientation_type=fixed",
        "alpha_SNR=0.9",
        "nnz=3",
        "seed=42",
    )
