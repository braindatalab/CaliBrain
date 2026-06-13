# =============================================================================
# BMN implementation
#   - BMN: fixed scalar noise variance
#   - BMN_joint: optional adaptive common noise learning
#
# Supports:
#   n_orient = 1  -> fixed (EEG or MEG)
#   n_orient = 2  -> reduced free MEG
#   n_orient = 3  -> free EEG
#
# Notes
# -----
# - The BMN solvers expect flattened leadfields:
#       fixed      : L shape (M, N)
#       free_meg   : L shape (M, 2N)
#       free_eeg   : L shape (M, 3N)
#
# - The corresponding simulator / leadfield module should provide:
#       SourceSimulator
#       SensorSimulator
#       extract_subset_leadfield
#       lift_reduced_sources_to_3d
# =============================================================================

import logging
from typing import Optional, Dict, Any, Tuple

import numpy as np
from scipy.sparse import block_diag


# =============================================================================
# Utilities
# =============================================================================
def _symmetrize(A: np.ndarray) -> np.ndarray:
    """Return the symmetric part of a square matrix."""
    return 0.5 * (A + A.T)


def _svd_inverse(A: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """
    Stable matrix inverse using SVD.
    """
    U, S, Vt = np.linalg.svd(A, full_matrices=False)
    S_inv = 1.0 / np.maximum(S, eps)
    return U @ np.diag(S_inv) @ Vt


def _validate_bmn_inputs(
    L: np.ndarray,
    y: np.ndarray,
    n_orient: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Common validation for BMN / BMN_joint.

    Expected leadfield shapes
    -------------------------
    n_orient = 1: L is (M, N)
    n_orient = 2: L is (M, 2N)
    n_orient = 3: L is (M, 3N)
    """
    L = np.asarray(L, dtype=float)
    y = np.asarray(y, dtype=float)

    if y.ndim == 1:
        y = y[:, np.newaxis]

    if L.ndim != 2:
        raise ValueError("L must be 2D.")
    if y.ndim != 2:
        raise ValueError("y must have shape (M, T) or (M,).")
    if L.shape[0] != y.shape[0]:
        raise ValueError("L and y must have the same number of sensor rows.")
    if n_orient not in (1, 2, 3):
        raise ValueError("n_orient must be 1, 2, or 3.")

    if n_orient > 1 and (L.shape[1] % n_orient != 0):
        raise ValueError(
            f"For n_orient={n_orient}, L must have k*N columns with k={n_orient}."
        )

    return L, y


# =============================================================================
# sLORETA normalization
# =============================================================================
def compute_W(L: np.ndarray, n_orient: int = 1, beta: float = 1e-6) -> np.ndarray:
    """
    Compute sLORETA-type normalization matrix W.

    Supports
    --------
    1) Fixed orientation:
       - L shape: (M, N)
       - W shape: (N, N), diagonal

    2) Reduced / free orientation:
       - L shape: (M, kN), with k in {2, 3}
       - W shape: (kN, kN), block-diagonal with k x k blocks

    Notes
    -----
    - Uses SVD-based inversion for numerical stability.
    - Uses symmetrization before eigendecomposition.
    - For n_orient > 1, each source contributes one local k x k normalization block.
    """
    L = np.asarray(L, dtype=float)
    if L.ndim != 2:
        raise ValueError("L must be 2D.")
    if n_orient not in (1, 2, 3):
        raise ValueError("n_orient must be 1, 2, or 3.")

    M, dim = L.shape
    eps = 1e-12

    LLt = _symmetrize(L @ L.T)
    LLt_reg = _symmetrize(LLt + beta * np.eye(M))
    LLt_inv = _svd_inverse(LLt_reg, eps=eps)

    # -------------------------------------------------------------------------
    # Fixed orientation: scalar normalization per source
    # -------------------------------------------------------------------------
    if n_orient == 1:
        A = LLt_inv @ L
        diag_S = np.sum(L * A, axis=0)
        W_diag = 1.0 / np.sqrt(np.maximum(diag_S, eps))
        return np.diag(W_diag)

    # -------------------------------------------------------------------------
    # Reduced / free orientation: generic k x k local blocks, k in {2, 3}
    # -------------------------------------------------------------------------
    if dim % n_orient != 0:
        raise ValueError(
            f"Lead-field L must have {n_orient}N columns for n_orient={n_orient}."
        )

    N = dim // n_orient
    S_hat = _symmetrize(L.T @ LLt_inv @ L)

    W_blocks = []
    for n in range(N):
        sl = slice(n_orient * n, n_orient * (n + 1))
        S_n = _symmetrize(S_hat[sl, sl])

        evals, evecs = np.linalg.eigh(S_n)
        evals = np.maximum(evals, eps)

        # W_n = S_n^{-1/2}
        W_n = evecs @ np.diag(1.0 / np.sqrt(evals)) @ evecs.T
        W_n = _symmetrize(W_n)
        W_blocks.append(W_n)

    return block_diag(W_blocks, format="csr").toarray()


# =============================================================================
# BMN with fixed known noise variance
# =============================================================================
def BMN_bayesian_opt(
    y: np.ndarray,
    L: np.ndarray,
    alpha: float,
    maxit: int = 1000,
    tol: float = 1e-6,
    init_gamma: Optional[float] = None,
    logger: Optional[logging.Logger] = None,
    verbose: bool = False,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    BMN optimization using Bayesian evidence maximization for one common
    source variance.

    Model
    -----
        y_t ~ N(L x_t, alpha I)
        x_t ~ N(0, gamma I)

    Notes
    -----
    - This is the normal BMN without adaptive noise learning.
    - Gamma is a single common scalar variance in the internal optimization
      coordinate system.
    """
    L = np.asarray(L, dtype=float).copy()
    y = np.asarray(y, dtype=float).copy()

    gamma = 1.0 if init_gamma is None else float(init_gamma)
    eps = np.finfo(float).eps

    if gamma <= 0:
        raise ValueError("init_gamma must be positive.")

    if y.ndim == 1:
        y = y[:, np.newaxis]
    if y.ndim != 2:
        raise ValueError("y must have shape (M, T) or (M,).")
    if L.ndim != 2:
        raise ValueError("L must be 2D.")
    if L.shape[0] != y.shape[0]:
        raise ValueError("L and y must have the same number of sensor rows.")

    M, T = y.shape

    y_original_scale = np.linalg.norm(y, ord="fro")
    L_original_scale = np.linalg.norm(L, ord=2)

    if y_original_scale < eps or L_original_scale < eps:
        raise ValueError("Degenerate input: y or L has (near) zero norm.")

    # Scale-normalized optimization
    y = y / y_original_scale
    L = L / L_original_scale
    alpha = float(alpha) / (y_original_scale ** 2)

    if alpha <= 0:
        raise ValueError("alpha must be positive.")

    LLt = _symmetrize(L @ L.T)

    for it in range(maxit):
        gamma_old = gamma

        model_cov = _symmetrize(gamma * LLt + alpha * np.eye(M))
        model_cov_inv = _svd_inverse(model_cov, eps=eps)

        model_cov_inv_y = model_cov_inv @ y
        numerator = np.trace(y.T @ model_cov_inv @ LLt @ model_cov_inv_y) / T
        denominator = np.trace(model_cov_inv @ LLt)

        gamma = numerator / max(denominator, eps)
        gamma = max(float(gamma), eps)

        err = np.abs(gamma - gamma_old) / (np.abs(gamma_old) + eps)

        if verbose and logger is not None:
            logger.debug(f"BMN iter {it:4d}: gamma={gamma:.6e}, err={err:.3e}")

        if err < tol:
            break

    model_cov = _symmetrize(gamma * LLt + alpha * np.eye(M))
    model_cov_inv = _svd_inverse(model_cov, eps=eps)

    A = L.T @ model_cov_inv @ y
    x_est = gamma * A

    posterior_cov = gamma * np.eye(L.shape[1]) - gamma**2 * (L.T @ model_cov_inv @ L)
    posterior_cov = _symmetrize(posterior_cov)

    # Map posterior outputs back to original coefficient scale
    scale_factor = y_original_scale / L_original_scale
    x_hat = scale_factor * x_est
    posterior_cov = (scale_factor ** 2) * posterior_cov
    posterior_cov = _symmetrize(posterior_cov)

    # gamma is returned as the internal common scalar hyperparameter
    return x_hat, posterior_cov, float(gamma)


def BMN(
    L: np.ndarray,
    y: np.ndarray,
    noise_var: float,
    n_orient: int = 1,
    max_iter: int = 1000,
    tol: float = 1e-6,
    init_gamma: Optional[float] = None,
    verbose: bool = False,
    normalization: bool = False,
    logger: Optional[logging.Logger] = None,
) -> Dict[str, Any]:
    """
    BMN estimate with optional sLORETA normalization.

    Supports
    --------
    n_orient = 1  -> fixed (EEG or MEG)
    n_orient = 2  -> reduced free MEG
    n_orient = 3  -> free EEG

    Notes
    -----
    - `posterior_mean` and `posterior_cov` are returned in the original
      coefficient space.
    - `gamma` is the learned common scalar hyperparameter in the internal
      optimization parameterization, so it should be treated mainly as a
      diagnostic quantity, especially when `normalization=True`.
    """
    L, y = _validate_bmn_inputs(L=L, y=y, n_orient=n_orient)

    noise_var = float(noise_var)
    if noise_var <= 0:
        raise ValueError("noise_var must be positive.")

    M = L.shape[0]

    # Optional sLORETA normalization
    if normalization:
        W = compute_W(L, n_orient=n_orient, beta=1e-6)
        L_normal = L @ W
    else:
        W = np.eye(L.shape[1])
        L_normal = L

    # Fixed known-noise whitening
    whitener = (1.0 / np.sqrt(noise_var)) * np.eye(M)
    y_white = whitener @ y
    L_white = whitener @ L_normal

    x_hat_normal, posterior_cov_normal, gamma = BMN_bayesian_opt(
        y=y_white,
        L=L_white,
        alpha=1.0,  # after whitening, noise covariance is I
        maxit=max_iter,
        tol=tol,
        init_gamma=init_gamma,
        logger=logger,
        verbose=verbose,
    )

    # Undo normalization
    x_hat = W @ x_hat_normal
    posterior_cov = W @ posterior_cov_normal @ W.T
    posterior_cov = _symmetrize(posterior_cov)

    n_coeff = posterior_cov.shape[0]
    n_sources = L.shape[1] if n_orient == 1 else L.shape[1] // n_orient

    out = {
        "posterior_mean": x_hat,
        "posterior_cov": posterior_cov,
        "noise_var": float(noise_var),
        "gamma": float(gamma),
        "coefficient_indices": np.arange(n_coeff),
        "source_indices": np.arange(n_sources),
        "active_indices": np.arange(n_coeff),  # backward-compat alias (coefficient-level)
    }

    # Generic reshape for n_orient = 2 or 3
    if n_orient > 1:
        out["posterior_mean_reshaped"] = x_hat.reshape(n_sources, n_orient, x_hat.shape[1])

    return out


# =============================================================================
# Common-noise update for joint BMN
# =============================================================================
def update_common_lambda_convex(
    y: np.ndarray,
    L: np.ndarray,
    posterior_mean: np.ndarray,
    C_inv: np.ndarray,
    eps: float = 1e-12,
) -> float:
    """
    Convex-bounding update rule for one common scalar noise variance in
    normalized scale.

    Returns
    -------
    alpha_new, where alpha = lambda / ||Y||_F^2
    """
    _, T = y.shape

    residual = y - L @ posterior_mean
    residual_term = (np.linalg.norm(residual, ord="fro") ** 2) / T
    denominator = np.trace(C_inv)

    alpha_new = np.sqrt(max(residual_term, 0.0) / max(denominator, eps))
    return max(float(alpha_new), eps)


# =============================================================================
# Joint BMN optimization with optional adaptive common noise learning
# =============================================================================
def BMN_joint_opt(
    y: np.ndarray,
    L: np.ndarray,
    noise_var: Optional[float] = None,
    maxit: int = 10000,
    tol: float = 1e-6,
    init_gamma: Optional[float] = None,
    init_lambda: Optional[float] = None,
    learn_noise: bool = False,
    logger: Optional[logging.Logger] = None,
    verbose: bool = False,
    track_history: bool = True,
) -> Tuple[np.ndarray, np.ndarray, float, float, Dict[str, list]]:
    """
    BMN optimization with one common source variance gamma and optional
    one common scalar sensor noise variance lambda.

    Model
    -----
        y_t ~ N(L x_t, lambda I)
        x_t ~ N(0, gamma I)

    Notes
    -----
    - Gamma update is the same scalar BMN update rule as in BMN_bayesian_opt.
    - Lambda update is convex-bounding only.
    - Convergence is checked on gamma.
    - Returned `gamma` is the internal common scalar hyperparameter in the
      optimization coordinate system.
    """
    L = np.asarray(L, dtype=float).copy()
    y = np.asarray(y, dtype=float).copy()

    eps = np.finfo(float).eps
    gamma = 1.0 if init_gamma is None else float(init_gamma)
    gamma = max(gamma, eps)

    if y.ndim == 1:
        y = y[:, np.newaxis]
    if y.ndim != 2:
        raise ValueError("y must have shape (M, T) or (M,).")
    if L.ndim != 2:
        raise ValueError("L must be 2D.")
    if L.shape[0] != y.shape[0]:
        raise ValueError("L and y must have the same number of sensor rows.")

    M, T = y.shape

    y_original_scale = np.linalg.norm(y, ord="fro")
    L_original_scale = np.linalg.norm(L, ord=2)

    if y_original_scale < eps or L_original_scale < eps:
        raise ValueError("Degenerate input: y or L has (near) zero norm.")

    # Scale-normalized optimization
    y = y / y_original_scale
    L = L / L_original_scale

    if learn_noise:
        if noise_var is not None:
            raise ValueError(
                "When learn_noise=True, noise_var must be None. "
                "Use init_lambda for initialization."
            )
        lambda_var = 1.0 if init_lambda is None else float(init_lambda)
        if lambda_var <= 0:
            raise ValueError("init_lambda must be positive when learn_noise=True.")
    else:
        if noise_var is None:
            raise ValueError("When learn_noise=False, noise_var must be provided.")
        lambda_var = float(noise_var)
        if lambda_var <= 0:
            raise ValueError("noise_var must be positive.")

    alpha = lambda_var / (y_original_scale ** 2)
    LLt = _symmetrize(L @ L.T)

    hist: Dict[str, list] = {}
    if track_history:
        hist["gamma_hist"] = []
        hist["lambda_hist"] = []
        hist["noise_var_hist"] = []
        hist["err_gamma_hist"] = []

    for it in range(maxit):
        gamma_old = gamma
        alpha_old = alpha

        model_cov = _symmetrize(gamma_old * LLt + alpha_old * np.eye(M))
        model_cov_inv = _svd_inverse(model_cov, eps)

        # Same scalar gamma update as normal BMN
        model_cov_inv_y = model_cov_inv @ y
        numerator = np.trace(y.T @ model_cov_inv @ LLt @ model_cov_inv_y) / T
        denominator = np.trace(model_cov_inv @ LLt)

        gamma = numerator / max(denominator, eps)
        gamma = max(float(gamma), eps)

        # Optional common lambda update
        if learn_noise:
            model_cov = _symmetrize(gamma * LLt + alpha_old * np.eye(M))
            model_cov_inv = _svd_inverse(model_cov, eps)

            A = L.T @ model_cov_inv @ y
            x_est_norm = gamma * A

            alpha = update_common_lambda_convex(
                y=y,
                L=L,
                posterior_mean=x_est_norm,
                C_inv=model_cov_inv,
                eps=eps,
            )
        else:
            alpha = alpha_old

        err_gamma = np.abs(gamma - gamma_old) / (np.abs(gamma_old) + eps)
        lambda_curr = alpha * (y_original_scale ** 2)

        if track_history:
            hist["gamma_hist"].append(float(gamma))
            hist["lambda_hist"].append(float(lambda_curr))
            hist["noise_var_hist"].append(float(lambda_curr))
            hist["err_gamma_hist"].append(float(err_gamma))

        if verbose and logger is not None:
            logger.debug(
                f"BMN iter {it:4d}: gamma={gamma:.6e}, "
                f"lambda={lambda_curr:.6e}, err_gamma={err_gamma:.3e}"
            )

        if err_gamma < tol:
            break

    model_cov = _symmetrize(gamma * LLt + alpha * np.eye(M))
    model_cov_inv = _svd_inverse(model_cov, eps)

    A = L.T @ model_cov_inv @ y
    x_est = gamma * A

    posterior_cov = gamma * np.eye(L.shape[1]) - gamma**2 * (L.T @ model_cov_inv @ L)
    posterior_cov = _symmetrize(posterior_cov)

    # Map posterior outputs back to original coefficient scale
    scale_factor = y_original_scale / L_original_scale
    x_hat = scale_factor * x_est
    posterior_cov = (scale_factor ** 2) * posterior_cov
    posterior_cov = _symmetrize(posterior_cov)

    lambda_var = alpha * (y_original_scale ** 2)

    return x_hat, posterior_cov, float(gamma), float(lambda_var), hist


# =============================================================================
# Public joint BMN API
# =============================================================================
def BMN_joint(
    L: np.ndarray,
    y: np.ndarray,
    noise_var: Optional[float] = None,
    n_orient: int = 1,
    max_iter: int = 1000,
    tol: float = 1e-6,
    init_gamma: Optional[float] = None,
    init_lambda: Optional[float] = None,
    learn_noise: bool = False,
    verbose: bool = False,
    normalization: bool = False,
    track_history: bool = True,
    logger: Optional[logging.Logger] = None,
    **kwargs,
) -> Dict[str, Any]:
    """
    BMN estimate with optional sLORETA normalization and optional adaptive
    common-noise learning.

    Supports
    --------
    n_orient = 1  -> fixed (EEG or MEG)
    n_orient = 2  -> reduced free MEG
    n_orient = 3  -> free EEG

    Notes
    -----
    - `posterior_mean` and `posterior_cov` are returned in the original
      coefficient space.
    - `gamma` is the learned common scalar hyperparameter in the internal
      optimization parameterization, so it should be treated mainly as a
      diagnostic quantity, especially when `normalization=True`.
    """
    L, y = _validate_bmn_inputs(L=L, y=y, n_orient=n_orient)

    if learn_noise:
        if noise_var is not None:
            raise ValueError(
                "When learn_noise=True, noise_var must be None. "
                "Use init_lambda for initialization."
            )
    else:
        if noise_var is None:
            raise ValueError("When learn_noise=False, noise_var must be provided.")
        if float(noise_var) <= 0:
            raise ValueError("noise_var must be positive.")

    # Optional sLORETA normalization
    if normalization:
        W = compute_W(L, n_orient=n_orient, beta=1e-6)
        L_normal = L @ W
    else:
        W = np.eye(L.shape[1])
        L_normal = L

    x_hat_normal, posterior_cov_normal, gamma, lambda_var, hist = BMN_joint_opt(
        y=y,
        L=L_normal,
        noise_var=noise_var,
        maxit=max_iter,
        tol=tol,
        init_gamma=init_gamma,
        init_lambda=init_lambda,
        learn_noise=learn_noise,
        logger=logger,
        verbose=verbose,
        track_history=track_history,
    )

    # Undo normalization
    x_hat = W @ x_hat_normal
    posterior_cov = W @ posterior_cov_normal @ W.T
    posterior_cov = _symmetrize(posterior_cov)

    n_coeff = posterior_cov.shape[0]
    n_sources = L.shape[1] if n_orient == 1 else L.shape[1] // n_orient

    out = {
        "posterior_mean": x_hat,
        "posterior_cov": posterior_cov,
        "gamma": float(gamma),
        "lambda": float(lambda_var),
        "noise_var": float(lambda_var),   # compatibility alias
        "coefficient_indices": np.arange(n_coeff),
        "source_indices": np.arange(n_sources),
        "active_indices": np.arange(n_coeff),  # backward-compat alias (coefficient-level)
    }

    # Generic reshape for n_orient = 2 or 3
    if n_orient > 1:
        out["posterior_mean_reshaped"] = x_hat.reshape(n_sources, n_orient, x_hat.shape[1])

    if track_history:
        out.update(hist)

    return out


# =============================================================================
# Example usage
# =============================================================================
# Assumes the following are already available from your simulation / forward module:
#   SourceSimulator
#   SensorSimulator
#   extract_subset_leadfield
#   lift_reduced_sources_to_3d

FWD_FIF = r"C:\Users\User\CaliBrain\calibrain\CC120264-fwd.fif"

src_sim = SourceSimulator()
sen_sim = SensorSimulator()

# =============================================================================
# 1) Fixed-orientation MEG using BMN (known noise variance)
#    k = 1, n_orient = 1
# =============================================================================
lf_fixed_meg = extract_subset_leadfield(
    fwd_fif=FWD_FIF,
    setting="fixed",
    coil_name="mag",
    n_per_hemi=642,
    seed=0,
    scale_to_nAm=True,
)

x_fixed_meg, active_fixed_meg = src_sim.simulate(
    setting="fixed",
    n_sources=lf_fixed_meg["n_sources"],
    nnz=10,
    seed=43,
)

y_clean_fixed_meg, y_noisy_fixed_meg, noise_fixed_meg, eta_fixed_meg = sen_sim.simulate(
    x=x_fixed_meg,
    L=lf_fixed_meg["L_flat"],   # (M, N)
    alpha_SNR=0.5,
    sensor_white_noise_std=1.0,
    seed=43,
)

noise_var_fixed_meg = float(np.var(noise_fixed_meg))

out_fixed_bmn = BMN(
    L=lf_fixed_meg["L_flat"],
    y=y_noisy_fixed_meg,
    noise_var=noise_var_fixed_meg,
    n_orient=1,
    normalization=True,
    max_iter=1000,
    tol=1e-6,
)

print("Fixed MEG BMN posterior_mean shape:", out_fixed_bmn["posterior_mean"].shape)
print("Fixed MEG BMN posterior_cov shape :", out_fixed_bmn["posterior_cov"].shape)
print("Fixed MEG BMN gamma               :", out_fixed_bmn["gamma"])


# =============================================================================
# 2) Reduced free-MEG using BMN (known noise variance)
#    k = 2, n_orient = 2
# =============================================================================
lf_free_meg = extract_subset_leadfield(
    fwd_fif=FWD_FIF,
    setting="free_meg",
    coil_name="mag",
    n_per_hemi=642,
    seed=0,
    scale_to_nAm=True,
)

a_free_meg, active_free_meg = src_sim.simulate(
    setting="free_meg",
    n_sources=lf_free_meg["n_sources"],
    nnz=10,
    seed=43,
)

y_clean_free_meg, y_noisy_free_meg, noise_free_meg, eta_free_meg = sen_sim.simulate(
    x=a_free_meg,
    L=lf_free_meg["L_block"],   # (M, N, 2)
    alpha_SNR=0.5,
    sensor_white_noise_std=1.0,
    seed=43,
)

noise_var_free_meg = float(np.var(noise_free_meg))

out_free_meg_bmn = BMN(
    L=lf_free_meg["L_flat"],    # (M, 2N)
    y=y_noisy_free_meg,
    noise_var=noise_var_free_meg,
    n_orient=2,
    normalization=True,
    max_iter=1000,
    tol=1e-6,
)

print("Free MEG BMN posterior_mean shape          :", out_free_meg_bmn["posterior_mean"].shape)
print("Free MEG BMN posterior_mean_reshaped shape :", out_free_meg_bmn["posterior_mean_reshaped"].shape)
print("Free MEG BMN posterior_cov shape           :", out_free_meg_bmn["posterior_cov"].shape)
print("Free MEG BMN gamma                         :", out_free_meg_bmn["gamma"])

# Optional: lift reduced posterior mean back to the retained local 3D basis
xhat_free_meg_3d = lift_reduced_sources_to_3d(
    out_free_meg_bmn["posterior_mean_reshaped"],
    lf_free_meg["Q_basis"],
)
print("Lifted free MEG posterior mean 3D shape:", xhat_free_meg_3d.shape)


# =============================================================================
# 3) Fixed-orientation MEG using BMN_joint (adaptive common noise learning)
#    k = 1, n_orient = 1
# =============================================================================
out_fixed_joint = BMN_joint(
    L=lf_fixed_meg["L_flat"],
    y=y_noisy_fixed_meg,
    noise_var=None,                     # must be None when learn_noise=True
    n_orient=1,
    normalization=True,
    learn_noise=True,
    init_lambda=None,
    max_iter=1000,
    tol=1e-6,
    track_history=True,
)

print("Fixed MEG joint posterior_mean shape:", out_fixed_joint["posterior_mean"].shape)
print("Fixed MEG joint posterior_cov shape :", out_fixed_joint["posterior_cov"].shape)
print("Fixed MEG joint gamma               :", out_fixed_joint["gamma"])
print("Fixed MEG joint lambda              :", out_fixed_joint["lambda"])


# =============================================================================
# 4) Reduced free-MEG using BMN_joint (adaptive common noise learning)
#    k = 2, n_orient = 2
# =============================================================================
out_free_meg_joint = BMN_joint(
    L=lf_free_meg["L_flat"],     # (M, 2N)
    y=y_noisy_free_meg,
    noise_var=None,              # must be None when learn_noise=True
    n_orient=2,
    normalization=True,
    learn_noise=True,
    init_lambda=None,
    max_iter=1000,
    tol=1e-6,
    track_history=True,
)

print("Free MEG joint posterior_mean shape          :", out_free_meg_joint["posterior_mean"].shape)
print("Free MEG joint posterior_mean_reshaped shape :", out_free_meg_joint["posterior_mean_reshaped"].shape)
print("Free MEG joint posterior_cov shape           :", out_free_meg_joint["posterior_cov"].shape)
print("Free MEG joint gamma                         :", out_free_meg_joint["gamma"])
print("Free MEG joint lambda                        :", out_free_meg_joint["lambda"])

# Optional: lift reduced posterior mean back to the retained local 3D basis
xhat_free_meg_joint_3d = lift_reduced_sources_to_3d(
    out_free_meg_joint["posterior_mean_reshaped"],
    lf_free_meg["Q_basis"],
)
print("Lifted free MEG joint posterior mean 3D shape:", xhat_free_meg_joint_3d.shape)


# =============================================================================
# 5) Fixed-orientation EEG using BMN
#    k = 1, n_orient = 1
#
# Your current forward file has 0 EEG channels, so this will not run with
# CC120264-fwd.fif. Use an EEG or EEG+MEG forward file.
# =============================================================================
# EEG_FWD_FIF = r"path_to_eeg_or_mixed_forward.fif"
#
# lf_fixed_eeg = extract_subset_leadfield(
#     fwd_fif=EEG_FWD_FIF,
#     setting="fixed",
#     coil_name="eeg",
#     n_per_hemi=642,
#     seed=0,
#     scale_to_nAm=True,
# )
#
# x_fixed_eeg, active_fixed_eeg = src_sim.simulate(
#     setting="fixed",
#     n_sources=lf_fixed_eeg["n_sources"],
#     nnz=10,
#     seed=43,
# )
#
# y_clean_fixed_eeg, y_noisy_fixed_eeg, noise_fixed_eeg, eta_fixed_eeg = sen_sim.simulate(
#     x=x_fixed_eeg,
#     L=lf_fixed_eeg["L_flat"],   # (M, N)
#     alpha_SNR=0.5,
#     sensor_white_noise_std=1.0,
#     seed=43,
# )
#
# out_fixed_eeg_bmn = BMN(
#     L=lf_fixed_eeg["L_flat"],
#     y=y_noisy_fixed_eeg,
#     noise_var=float(np.var(noise_fixed_eeg)),
#     n_orient=1,
#     normalization=True,
#     max_iter=1000,
#     tol=1e-6,
# )
#
# print("Fixed EEG BMN posterior_mean shape:", out_fixed_eeg_bmn["posterior_mean"].shape)


# =============================================================================
# 6) Free EEG using BMN
#    k = 3, n_orient = 3
# =============================================================================
# lf_free_eeg = extract_subset_leadfield(
#     fwd_fif=EEG_FWD_FIF,
#     setting="free_eeg",
#     coil_name="eeg",
#     n_per_hemi=642,
#     seed=0,
#     scale_to_nAm=True,
# )
#
# x_free_eeg, active_free_eeg = src_sim.simulate(
#     setting="free_eeg",
#     n_sources=lf_free_eeg["n_sources"],
#     nnz=10,
#     seed=43,
# )
#
# y_clean_free_eeg, y_noisy_free_eeg, noise_free_eeg, eta_free_eeg = sen_sim.simulate(
#     x=x_free_eeg,
#     L=lf_free_eeg["L_block"],   # (M, N, 3)
#     alpha_SNR=0.5,
#     sensor_white_noise_std=1.0,
#     seed=43,
# )
#
# out_free_eeg_bmn = BMN(
#     L=lf_free_eeg["L_flat"],    # (M, 3N)
#     y=y_noisy_free_eeg,
#     noise_var=float(np.var(noise_free_eeg)),
#     n_orient=3,
#     normalization=True,
#     max_iter=1000,
#     tol=1e-6,
# )
#
# print("Free EEG BMN posterior_mean_reshaped shape:", out_free_eeg_bmn["posterior_mean_reshaped"].shape)


# =============================================================================
# 7) Fixed-orientation EEG using BMN_joint
#    k = 1, n_orient = 1
# =============================================================================
# out_fixed_eeg_joint = BMN_joint(
#     L=lf_fixed_eeg["L_flat"],
#     y=y_noisy_fixed_eeg,
#     noise_var=None,
#     n_orient=1,
#     normalization=True,
#     learn_noise=True,
#     init_lambda=None,
#     max_iter=1000,
#     tol=1e-6,
#     track_history=True,
# )
#
# print("Fixed EEG joint posterior_mean shape:", out_fixed_eeg_joint["posterior_mean"].shape)
# print("Fixed EEG joint lambda:", out_fixed_eeg_joint["lambda"])


# =============================================================================
# 8) Free EEG using BMN_joint
#    k = 3, n_orient = 3
# =============================================================================
# out_free_eeg_joint = BMN_joint(
#     L=lf_free_eeg["L_flat"],    # (M, 3N)
#     y=y_noisy_free_eeg,
#     noise_var=None,
#     n_orient=3,
#     normalization=True,
#     learn_noise=True,
#     init_lambda=None,
#     max_iter=1000,
#     tol=1e-6,
#     track_history=True,
# )
#
# print("Free EEG joint posterior_mean_reshaped shape:", out_free_eeg_joint["posterior_mean_reshaped"].shape)
# print("Free EEG joint lambda:", out_free_eeg_joint["lambda"])


