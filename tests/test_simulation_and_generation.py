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


def _make_normals(n_sources, seed=0, scale=3.0):
    rng = np.random.RandomState(seed)
    normals = rng.randn(n_sources, 3)
    normals /= np.linalg.norm(normals, axis=1, keepdims=True)
    return normals * scale  # non-unit on purpose: simulator must normalize


def _make_q_basis(n_sources, seed=1):
    rng = np.random.RandomState(seed)
    Q = np.zeros((n_sources, 3, 2))
    for i in range(n_sources):
        Q[i], _ = np.linalg.qr(rng.randn(3, 2))
    return Q


def test_source_simulator_free_eeg_returns_rank_one_blocks_along_normal():
    simulator = SourceSimulator(ERP_config=_make_erp_config())
    normals = _make_normals(6)

    x, active_sources = simulator.simulate(
        n_sources=6,
        nnz=2,
        orientation_type="free",
        coil_type=FIFF.FIFFV_COIL_EEG,
        seed=11,
        cortical_normals=normals,
    )

    assert x.shape == (6, 3, 250)
    assert len(active_sources) == 2
    assert np.count_nonzero(np.linalg.norm(x.reshape(6, -1), axis=1)) == 2

    unit = normals / np.linalg.norm(normals, axis=1, keepdims=True)
    for idx in active_sources:
        block = x[idx]
        assert np.linalg.matrix_rank(block) == 1
        # every time sample is parallel to the unit normal
        coeff = unit[idx] @ block
        np.testing.assert_allclose(block, np.outer(unit[idx], coeff), atol=1e-10)


def test_source_simulator_free_meg_projects_normal_through_q_basis():
    simulator = SourceSimulator(ERP_config=_make_erp_config())
    normals = _make_normals(5)
    Q = _make_q_basis(5)

    x, active_sources = simulator.simulate(
        n_sources=5,
        nnz=2,
        orientation_type="free",
        coil_type=FIFF.FIFFV_COIL_VV_MAG_T1,
        seed=13,
        cortical_normals=normals,
        Q_basis=Q,
    )

    assert x.shape == (5, 2, 250)
    assert len(active_sources) == 2
    assert np.count_nonzero(np.linalg.norm(x.reshape(5, -1), axis=1)) == 2

    unit = normals / np.linalg.norm(normals, axis=1, keepdims=True)
    for idx in active_sources:
        direction = Q[idx].T @ unit[idx]  # not renormalized
        block = x[idx]
        assert np.linalg.matrix_rank(block) == 1
        coeff = direction @ block / (direction @ direction)
        np.testing.assert_allclose(block, np.outer(direction, coeff), atol=1e-10)


def test_source_simulator_free_same_waveform_for_eeg_and_meg():
    simulator = SourceSimulator(ERP_config=_make_erp_config())
    normals = _make_normals(5)
    Q = _make_q_basis(5)

    x3, act3 = simulator.simulate(
        n_sources=5, nnz=2, orientation_type="free",
        coil_type=FIFF.FIFFV_COIL_EEG, seed=3, cortical_normals=normals,
    )
    x2, act2 = simulator.simulate(
        n_sources=5, nnz=2, orientation_type="free",
        coil_type=FIFF.FIFFV_COIL_VV_MAG_T1, seed=3,
        cortical_normals=normals, Q_basis=Q,
    )

    np.testing.assert_array_equal(act3, act2)
    unit = normals / np.linalg.norm(normals, axis=1, keepdims=True)
    for idx in act3:
        w3 = unit[idx] @ x3[idx]
        d = Q[idx].T @ unit[idx]
        w2 = d @ x2[idx] / (d @ d)
        np.testing.assert_allclose(w3, w2, atol=1e-10)


def test_source_simulator_free_requires_normals_and_basis():
    simulator = SourceSimulator(ERP_config=_make_erp_config())

    with pytest.raises(ValueError, match="cortical_normals"):
        simulator.simulate(
            n_sources=4, nnz=1, orientation_type="free",
            coil_type=FIFF.FIFFV_COIL_EEG, seed=1,
        )
    with pytest.raises(ValueError, match="Q_basis"):
        simulator.simulate(
            n_sources=4, nnz=1, orientation_type="free",
            coil_type=FIFF.FIFFV_COIL_VV_MAG_T1, seed=1,
            cortical_normals=_make_normals(4),
        )


def test_source_simulator_free_validates_inputs():
    simulator = SourceSimulator(ERP_config=_make_erp_config())
    kw = dict(n_sources=4, nnz=1, orientation_type="free", seed=1)

    with pytest.raises(ValueError, match="shape"):
        simulator.simulate(coil_type=FIFF.FIFFV_COIL_EEG,
                           cortical_normals=np.ones((3, 3)), **kw)

    zero = _make_normals(4)
    zero[:] = 0.0
    with pytest.raises(ValueError, match="nonzero"):
        simulator.simulate(coil_type=FIFF.FIFFV_COIL_EEG,
                           cortical_normals=zero, **kw)

    bad_Q = _make_q_basis(4) * 2.0
    with pytest.raises(ValueError, match="orthonormal"):
        simulator.simulate(coil_type=FIFF.FIFFV_COIL_VV_MAG_T1,
                           cortical_normals=_make_normals(4), Q_basis=bad_Q, **kw)


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


def _make_generator_with_leadfield(tmp_path, leadfield_data):
    class StubLeadfieldBuilder:
        def get_leadfield(self, **kwargs):
            return leadfield_data

    return DataGenerator(
        solver=lambda **kwargs: None,
        solver_param_grid={"solver_name": ["dummy_solver"]},
        data_param_grid={"orientation_type": ["free"], "nnz": [2]},
        noise_param_grid={"noise_type": ["oracle"]},
        ERP_config=_make_erp_config(),
        source_simulator=SourceSimulator(ERP_config=_make_erp_config()),
        leadfield_builder=StubLeadfieldBuilder(),
        sensor_simulator=SensorSimulator(),
        posterior_dir=tmp_path / "posterior",
    )


_RUN_PARAMS = {
    "subject": "stub",
    "orientation_type": "free",
    "nnz": 2,
    "alpha_SNR": 0.5,
    "sensor_white_noise_std": 1.0,
}


@pytest.mark.parametrize("coil", ["eeg", "meg"])
def test_data_generator_prepare_run_data_free_uses_cortical_normals(tmp_path, coil):
    from calibrain.leadfield_builder import LeadfieldData

    n_sources, n_sensors = 6, 8
    normals = _make_normals(n_sources)
    rng = np.random.RandomState(0)
    if coil == "eeg":
        Q, k, coil_type = np.repeat(np.eye(3)[None], n_sources, axis=0), 3, FIFF.FIFFV_COIL_EEG
    else:
        Q, k, coil_type = _make_q_basis(n_sources), 2, FIFF.FIFFV_COIL_VV_MAG_T1

    leadfield_data = LeadfieldData(
        leadfield=rng.randn(n_sensors, n_sources, k),
        sensor_kind=1, sensor_units=112, sensor_unitmult=0,
        coil_type=coil_type, src_coords=rng.randn(n_sources, 3),
        Q_basis=Q, cortical_normals=normals,
    )
    generator = _make_generator_with_leadfield(tmp_path, leadfield_data)

    prepared = generator._prepare_run_data(_RUN_PARAMS, seed=5)

    x = prepared["x"]
    assert x.shape == (n_sources, k, 250)
    for idx in prepared["x_active_indices"]:
        assert np.linalg.matrix_rank(x[idx]) == 1


def test_data_generator_prepare_run_data_free_requires_cortical_normals(tmp_path):
    from calibrain.leadfield_builder import LeadfieldData

    rng = np.random.RandomState(0)
    leadfield_data = LeadfieldData(
        leadfield=rng.randn(8, 6, 3),
        sensor_kind=2, sensor_units=107, sensor_unitmult=0,
        coil_type=FIFF.FIFFV_COIL_EEG, src_coords=rng.randn(6, 3),
        Q_basis=np.repeat(np.eye(3)[None], 6, axis=0), cortical_normals=None,
    )
    generator = _make_generator_with_leadfield(tmp_path, leadfield_data)

    with pytest.raises(ValueError, match="cortical_normals"):
        generator._prepare_run_data(_RUN_PARAMS, seed=5)
