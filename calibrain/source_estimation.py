import logging
import numpy as np
from numpy.linalg import inv
from matplotlib import cm
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from scipy import linalg
from scipy.spatial.distance import pdist, squareform
from scipy.stats import chi2, norm
from scipy.linalg import sqrtm
from scipy.sparse import coo_matrix, csr_matrix, diags, eye, kron, issparse, block_diag, identity
from functools import partial
import mne
from mne.utils import sqrtm_sym, eigh
from mne.io.constants import FIFF
from typing import Optional, Dict, Any, Tuple

from calibrain.utils import get_data_path

# ===================
# GAMMA-MAP Functions
# ===================

def _validate_gamma_map_inputs(
    L: np.ndarray,
    y: np.ndarray,
    n_orient: int,
) -> Tuple[np.ndarray, np.ndarray]:
    L = np.asarray(L, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)

    if y.ndim == 1:
        y = y[:, np.newaxis]

    if L.ndim != 2:
        raise ValueError("L must be 2D.")
    if y.ndim != 2:
        raise ValueError("y must have shape (M,T) or (M,).")
    if L.shape[0] != y.shape[0]:
        raise ValueError("L and y must have the same number of sensor rows.")
    if n_orient not in (1, 2, 3):
        raise ValueError("n_orient must be 1, 2, or 3.")
    if L.shape[1] % n_orient != 0:
        raise ValueError(
            f"For n_orient={n_orient}, L must have k*N columns with k={n_orient}."
        )

    return L, y

def _prepare_init_gamma(
    n_coeffs: int,
    n_orient: int,
    init_gamma=None,
) -> np.ndarray:
    n_groups = n_coeffs // n_orient

    if init_gamma is None:
        gamma0 = np.ones(n_coeffs, dtype=np.float64)

    elif isinstance(init_gamma, (float, np.floating, int, np.integer)):
        gamma0 = np.full(n_coeffs, float(init_gamma), dtype=np.float64)

    elif isinstance(init_gamma, tuple) and len(init_gamma) == 2:
        gamma0 = np.linspace(init_gamma[0], init_gamma[1], num=n_coeffs).astype(np.float64)

    else:
        gamma0 = np.asarray(init_gamma, dtype=np.float64).ravel()

        if gamma0.size == n_groups:
            gamma0 = np.repeat(gamma0, n_orient)
        elif gamma0.size != n_coeffs:
            raise ValueError(
                f"init_gamma must have length {n_coeffs} or {n_groups}; got {gamma0.size}."
            )

    gamma0 = np.maximum(gamma0, 0.0)

    if n_orient > 1:
        gamma_group = gamma0.reshape(n_groups, n_orient).mean(axis=1)
        gamma0 = np.repeat(gamma_group, n_orient)

    return gamma0

def _gamma_map_opt(
    M: np.ndarray,
    G: np.ndarray,
    sigma_squared: float,
    *,
    maxit: int = 300,
    tol: float = 1e-6,
    update_mode: int = 2,
    group_size: int = 1,
    init_gamma: Optional[np.ndarray] = None,
    verbose: bool = False,
    logger: Optional[logging.Logger] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    if logger is None:
        logger = logging.getLogger(__name__)

    G = np.asarray(G, dtype=np.float64).copy()
    M = np.asarray(M, dtype=np.float64).copy()

    n_coeffs_total = G.shape[1]
    n_sensors, n_times = M.shape
    eps = np.finfo(float).eps

    if n_coeffs_total % group_size != 0:
        raise ValueError("Number of coefficients must be divisible by group_size.")

    if init_gamma is None:
        gamma = np.ones(n_coeffs_total, dtype=np.float64)
    else:
        gamma = np.asarray(init_gamma, dtype=np.float64).copy()
        if gamma.shape != (n_coeffs_total,):
            raise ValueError(
                f"init_gamma must have shape ({n_coeffs_total},), got {gamma.shape}"
            )

    M_norm_c = np.linalg.norm(M @ M.T, ord="fro")
    if M_norm_c <= 0:
        raise ValueError("Degenerate data: M has zero norm.")
    M /= np.sqrt(M_norm_c)
    sigma_squared /= M_norm_c

    G_norm_c = np.linalg.norm(G, ord=np.inf)
    if G_norm_c <= 0:
        raise ValueError("Degenerate leadfield: G has zero norm.")
    G /= G_norm_c

    active_indices = np.arange(n_coeffs_total, dtype=int)
    gammas_full_old = gamma.copy()

    A_last = np.zeros((0, n_times), dtype=np.float64)
    CMinv_last = np.eye(n_sensors, dtype=np.float64)
    G_last = np.zeros((n_sensors, 0), dtype=np.float64)
    gamma_last = np.zeros((0,), dtype=np.float64)
    active_indices_last = np.zeros((0,), dtype=int)

    last_size = -1
    it_used = 0

    for itno in range(maxit):
        it_used = itno + 1

        gamma = np.nan_to_num(gamma, nan=0.0, posinf=0.0, neginf=0.0)
        gamma = np.maximum(gamma, 0.0)

        n_groups_active = gamma.size // group_size
        gamma_group = gamma.reshape(n_groups_active, group_size).mean(axis=1)
        gmask_group = np.abs(gamma_group) > eps
        gmask_coeff = np.repeat(gmask_group, group_size)

        if not np.all(gmask_coeff):
            active_indices = active_indices[gmask_coeff]
            gamma = gamma[gmask_coeff]
            G = G[:, gmask_coeff]

        if gamma.size == 0:
            break

        CM = (G * gamma[np.newaxis, :]) @ G.T
        CM.flat[:: n_sensors + 1] += sigma_squared

        U, S, _ = linalg.svd(CM, full_matrices=False)
        S = S[np.newaxis, :]
        CMinv = (U / (S + eps)) @ U.T

        CMinvG = CMinv @ G
        A = CMinvG.T @ M

        if update_mode == 1:
            numer = gamma**2 * np.mean((A * A.conj()).real, axis=1)
            denom = gamma * np.sum(G * CMinvG, axis=0)
        elif update_mode == 2:
            numer = gamma * np.sqrt(np.mean((A * A.conj()).real, axis=1))
            denom = np.sum(G * CMinvG, axis=0)
        elif update_mode == 3:
            numer = gamma**2 * np.mean((A * A.conj()).real, axis=1) + gamma * (
                1.0 - gamma * np.sum(G * CMinvG, axis=0)
            )
            denom = None
        else:
            raise ValueError("update_mode must be 1, 2, or 3.")

        if group_size == 1:
            if denom is None:
                gamma = numer
            elif update_mode == 2:
                gamma = numer / np.sqrt(np.maximum(denom, eps))
            else:
                gamma = numer / np.maximum(denom, eps)
        else:
            numer_group = np.sum(numer.reshape(-1, group_size), axis=1)

            if denom is None:
                gamma_group_new = numer_group
            else:
                denom_group = np.sum(denom.reshape(-1, group_size), axis=1)
                if update_mode == 2:
                    gamma_group_new = numer_group / np.sqrt(np.maximum(denom_group, eps))
                else:
                    gamma_group_new = numer_group / np.maximum(denom_group, eps)

            gamma = np.repeat(gamma_group_new / group_size, group_size)

        gamma = np.maximum(gamma, 0.0)

        gammas_full = np.zeros(n_coeffs_total, dtype=np.float64)
        gammas_full[active_indices] = gamma

        err = np.sum(np.abs(gammas_full - gammas_full_old)) / np.sum(
            np.abs(gammas_full_old) + eps
        )
        gammas_full_old = gammas_full.copy()

        A_last = A.copy()
        CMinv_last = CMinv.copy()
        G_last = G.copy()
        gamma_last = gamma.copy()
        active_indices_last = active_indices.copy()

        breaking = (err < tol) or (gamma.size == 0)

        if (gamma.size != last_size) or breaking:
            if verbose:
                logger.debug(f"it={itno:4d} active={gamma.size:4d} err={err:0.3e}")
            last_size = gamma.size

        if breaking:
            break

    if active_indices_last.size == 0:
        x_active = np.zeros((0, n_times), dtype=np.float64)
        posterior_cov_active = np.zeros((0, 0), dtype=np.float64)
        gammas_full = np.zeros(n_coeffs_total, dtype=np.float64)
        return x_active, active_indices_last, posterior_cov_active, gammas_full, it_used

    n_const = np.sqrt(M_norm_c) / G_norm_c
    x_active = n_const * gamma_last[:, None] * A_last

    posterior_cov_active = (
        np.diag(gamma_last) - gamma_last[:, None] * (G_last.T @ CMinv_last @ G_last) * gamma_last
    )
    posterior_cov_active = (n_const**2) * _symmetrize(posterior_cov_active)

    gammas_full = np.zeros(n_coeffs_total, dtype=np.float64)
    gammas_full[active_indices_last] = gamma_last

    return x_active, active_indices_last, posterior_cov_active, gammas_full, it_used

def gamma_map(
    L: np.ndarray,
    y: np.ndarray,
    noise_var: float,
    n_orient: int = 1,
    max_iter: int = 300,
    tol: float = 1e-6,
    update_mode: int = 2,
    init_gamma=None,
    verbose: bool = False,
    logger: Optional[logging.Logger] = None,
    **kwargs,
) -> Dict[str, Any]:
    if logger is None:
        logger = logging.getLogger(__name__)

    L, y = _validate_gamma_map_inputs(L=L, y=y, n_orient=n_orient)

    noise_var = float(noise_var)
    if noise_var <= 0:
        raise ValueError("noise_var must be positive.")

    n_sensors, n_times = y.shape
    n_coeffs = L.shape[1]
    n_sources = n_coeffs // n_orient

    gamma0 = _prepare_init_gamma(
        n_coeffs=n_coeffs,
        n_orient=n_orient,
        init_gamma=init_gamma,
    )

    whitener = (1.0 / np.sqrt(noise_var)) * np.eye(n_sensors, dtype=np.float64)
    y_w = whitener @ y
    L_w = whitener @ L

    x_active, active_indices, posterior_cov_active, gammas_full, n_iter = _gamma_map_opt(
        M=y_w,
        G=L_w,
        sigma_squared=1.0,
        maxit=max_iter,
        tol=tol,
        update_mode=update_mode,
        group_size=n_orient,
        init_gamma=gamma0,
        verbose=verbose,
        logger=logger,
    )

    x_hat = np.zeros((n_coeffs, n_times), dtype=np.float64)
    posterior_cov = np.zeros((n_coeffs, n_coeffs), dtype=np.float64)

    if active_indices.size > 0:
        x_hat[active_indices] = x_active
        posterior_cov[np.ix_(active_indices, active_indices)] = posterior_cov_active

    posterior_cov = _symmetrize(posterior_cov)

    out = {
        "posterior_mean": x_hat,
        "posterior_cov": posterior_cov,
        "posterior_cov_active": posterior_cov_active,
        "noise_var": float(noise_var),
        "gamma": float(np.mean(gammas_full)),
        "gammas_full": gammas_full,
        "active_indices": active_indices,
        "active_source_indices": np.unique(active_indices // n_orient),
        "coefficient_indices": np.arange(n_coeffs),
        "source_indices": np.arange(n_sources),
        "n_orient": int(n_orient),
        "n_iter": int(n_iter),
    }

    if n_orient > 1:
        out["posterior_mean_reshaped"] = x_hat.reshape(n_sources, n_orient, n_times)

    return out

# ==================
# sFlex Functions
# ==================

# def get_subset_source_rr_from_extract(lf_dict: Dict[str, Any]) -> np.ndarray:
#     src = lf_dict["fwd"]["src"]
#     rr_lh = src[0]["rr"][src[0]["vertno"]]
#     rr_rh = src[1]["rr"][src[1]["vertno"]]
#     rr_full = np.vstack([rr_lh, rr_rh])
#     subset_idx = np.asarray(lf_dict["subset_idx"], dtype=int)
#     return rr_full[subset_idx]

def compute_B(
    sigma: float,
    threshold_factor: float = 3.0,
    normalize: Optional[str] = "sym",
    eps: float = 1e-12,
    src_coords: np.ndarray = None,
):
    
    if src_coords.ndim != 2 or src_coords.shape[1] != 3:
        raise ValueError("src_coords must have shape (N,3).")

    if sigma <= 0:
        raise ValueError("sigma must be positive.")
    
    N = src_coords.shape[0]
    dist2 = squareform(pdist(src_coords, metric="sqeuclidean"))
    r2 = (threshold_factor * sigma) ** 2

    mask = dist2 <= r2
    rows, cols = np.nonzero(mask)
    weights = np.exp(-dist2[rows, cols] / (2.0 * sigma**2))

    B = coo_matrix((weights, (rows, cols)), shape=(N, N)).tocsr()
    B = 0.5 * (B + B.T)

    if normalize is None:
        return B

    row_sums = np.asarray(B.sum(axis=1)).ravel()

    if normalize == "row":
        inv = 1.0 / np.maximum(row_sums, eps)
        return diags(inv) @ B

    if normalize == "sym":
        inv_sqrt = 1.0 / np.sqrt(np.maximum(row_sums, eps))
        Dm = diags(inv_sqrt)
        B = Dm @ B @ Dm
        B = 0.5 * (B + B.T)
        return B

    raise ValueError("normalize must be None, 'row', or 'sym'.")

def _expand_spatial_basis(B, n_sources: int, n_orient: int):
    if issparse(B):
        B = B.tocsr()
    else:
        B = np.asarray(B, dtype=np.float64)

    if B.shape != (n_sources, n_sources):
        raise ValueError(
            f"B must have shape ({n_sources},{n_sources}); got {B.shape}."
        )

    if n_orient == 1:
        return B

    I_k = eye(n_orient, format="csr")
    if issparse(B):
        return kron(B, I_k, format="csr")
    return np.kron(B, np.eye(n_orient, dtype=np.float64))

def _right_multiply_dense_by_sparse(A: np.ndarray, S) -> np.ndarray:
    out = (S.T @ A.T).T
    return np.asarray(out, dtype=np.float64)

def gamma_map_sflex(
    L: np.ndarray,
    y: np.ndarray,
    noise_var: float,
    n_orient: int = 1,
    max_iter: int = 300,
    tol: float = 1e-6,
    update_mode: int = 2,
    init_gamma=None,
    sigma: float = 0.01,
    threshold_factor: float = 3.0,
    normalize: Optional[str] = "sym",
    eps: float = 1e-12,
    verbose: bool = False,
    logger: Optional[logging.Logger] = None,
    **kwargs,
) -> Dict[str, Any]:
    if logger is None:
        logger = logging.getLogger(__name__)

    L, y = _validate_gamma_map_inputs(L=L, y=y, n_orient=n_orient)

    n_coeffs = L.shape[1]
    n_sources = n_coeffs // n_orient
    n_times = y.shape[1]
    
    B = compute_B(
        sigma=sigma,
        threshold_factor=threshold_factor,
        normalize=normalize,
        eps=eps,
        src_coords=kwargs.get("src_coords"),
    )

    B_big = _expand_spatial_basis(B=B, n_sources=n_sources, n_orient=n_orient)

    if issparse(B_big):
        G = _right_multiply_dense_by_sparse(L, B_big)
    else:
        G = L @ B_big

    res_c = gamma_map(
        L=G,
        y=y,
        noise_var=noise_var,
        n_orient=n_orient,
        max_iter=max_iter,
        tol=tol,
        update_mode=update_mode,
        init_gamma=init_gamma,
        verbose=verbose,
        logger=logger,
    )

    c_hat = np.asarray(res_c["posterior_mean"], dtype=np.float64)
    Sigma_c = np.asarray(res_c["posterior_cov"], dtype=np.float64)

    if issparse(B_big):
        x_hat = np.asarray(B_big @ c_hat, dtype=np.float64)
        B_big_dense = B_big.toarray()
    else:
        B_big_dense = np.asarray(B_big, dtype=np.float64)
        x_hat = B_big_dense @ c_hat

    posterior_cov = B_big_dense @ Sigma_c @ B_big_dense.T
    posterior_cov = _symmetrize(posterior_cov)

    out = {
        "posterior_mean": x_hat,
        "posterior_cov": posterior_cov,
        "posterior_mean_coeff": c_hat,
        "posterior_cov_coeff": Sigma_c,
        "noise_var": float(res_c["noise_var"]),
        "gamma": float(res_c["gamma"]),
        "gammas_full": np.asarray(res_c["gammas_full"], dtype=np.float64),
        "active_indices": np.asarray(res_c["active_indices"], dtype=int),
        "active_source_indices": np.asarray(res_c["active_source_indices"], dtype=int),
        "coefficient_indices": np.arange(n_coeffs),
        "source_indices": np.arange(n_sources),
        "n_orient": int(n_orient),
        "n_iter": int(res_c["n_iter"]),
        "B_spatial": B,
    }

    if n_orient > 1:
        out["posterior_mean_reshaped"] = x_hat.reshape(n_sources, n_orient, n_times)

    return out

# ===================
# eLORETA Functions
# ===================

def sqrtm_sym(M, inv=False):
    """
    Compute the square root (or inverse square root) of symmetric matrices, 
    handling both 2D and block-diagonal 3D cases.
    """
    if M.ndim == 3:
        # Process each block separately (n_blocks, n, n)
        n_blocks, n, _ = M.shape
        S = np.zeros_like(M)
        s = np.zeros((n_blocks, n))
        for i in range(n_blocks):
            s_i, U_i = eigh(M[i])
            s_i = np.clip(s_i, 0, None)
            if inv:
                s_i = 1.0 / np.sqrt(s_i + np.finfo(float).eps)
            else:
                s_i = np.sqrt(s_i)
            S[i] = (U_i * s_i) @ U_i.T
            s[i] = s_i
        return S, s
    else:
        # Original 2D case
        s, U = eigh(M)
        s = np.clip(s, 0, None)
        if inv:
            s = 1.0 / np.sqrt(s + np.finfo(float).eps)
        else:
            s = np.sqrt(s)
        S = (U * s) @ U.T
        return S, s

def normalize_R(G, R, G_3, n_nzero, force_equal, n_src, n_orient):
    """
    Normalize the source covariance matrix (R) for consistency with eigenvalues.
    
    This function normalizes the product G @ R @ G.T so that its trace matches a
    reference value (n_nzero).
    
    Parameters
    ----------
    G : ndarray, shape (n_chan, n_src * n_orient)
        The lead-field or forward matrix after applying whitening and source scaling.
    R : ndarray
        The source covariance matrix; may be a 1D vector (single orientation) or
        a block diagonal structure (multiple orientations).
    G_3 : ndarray or None
        Reshaped version of G for multi-orientation sources (n_src x n_orient x n_chan),
        or None for single orientation.
    n_nzero : int
        The number of non-zero sensor dimensions (typically, the number of sensors).
    force_equal : bool
        If True, enforce equal orientation weights (i.e., treat sources with single orientation).
    n_src : int
        Number of sources (after accounting for orientation).
    n_orient : int
        Number of orientations per source (1 for fixed, 3 for free orientation).
        
    Returns
    -------
    G_R_Gt : ndarray
        The normalized product G @ R @ G.T.
    """
    # If sources are scalar (single orientation) or forced to have equal orientation,
    # perform element-wise multiplication for R.
    if n_orient == 1 or force_equal:  
        # R[:, np.newaxis] makes R a column vector, then multiply each column of G.T
        R_Gt = R[:, np.newaxis] * G.T  
    else:
        # For multi-orientation: Perform matrix multiplication with reshaped G.
        R_Gt = np.matmul(R, G_3).reshape(n_src * 3, -1)
    
    # Compute product G @ R @ G.T (the sensor-level covariance after applying R)
    G_R_Gt = G @ R_Gt
    # Compute the normalization factor as the trace divided by number of sensors (n_nzero)
    norm = np.trace(G_R_Gt) / n_nzero
    # Scale the matrix and R by the normalization factor
    G_R_Gt /= norm
    R /= norm
    return G_R_Gt

def get_G_3(G, n_orient):
    """
    Reshape and transpose the lead-field matrix G for multi-orientation sources.
    
    Parameters
    ----------
    G : ndarray, shape (n_chan, n_src * n_orient)
        The original lead-field matrix, after whitening and orientation‐prior scaling.
    n_orient : int
        Number of orientations per source (1 for fixed, 3 for free orientation).
        
    Returns
    -------
    ndarray or None :
        If n_orient > 1, returns an array of shape (n_src, n_orient, n_chan),
        so that each source’s 3×n_chan lead-field slice is one block.
        If n_orient == 1, returns None.
    """
    if n_orient == 1:
        return None  # No multi-orientation; nothing to reshape
    else:
        # 1) G originally is (n_chan, n_src * n_orient).
        #    We want to group every 'n_orient' columns into one source.
        # 2) First reshape to (n_chan, n_src, n_orient):
        #       G.reshape(n_chan, n_src, n_orient)
        # 3) Then transpose axes so that the block for source i is at G_3[i]:
        #       .transpose(1, 2, 0)  →  (n_src, n_orient, n_chan)
        return G.reshape(G.shape[0], -1, n_orient).transpose(1, 2, 0)

def R_sqrt_mult(other, R_sqrt):
    """
    Efficiently compute the multiplication: other @ R_sqrt.
    
    This function handles both diagonal and block-diagonal cases for R_sqrt.
    
    Parameters
    ----------
    other : ndarray, shape (n_chan, n_src * n_orient) or similar
        The matrix to be multiplied with R_sqrt.
    R_sqrt : ndarray
        The square root of the source covariance matrix R. It is either a 1D vector
        (for a diagonal matrix) or a 3D array (for block-diagonal multi-orientation case).
        
    Returns
    -------
    out : ndarray
        The result of the matrix multiplication.
    """
    if R_sqrt.ndim == 1:  # Diagonal matrix represented as a vector
        # Ensure compatible dimensions: other.shape[1] == size of R_sqrt
        assert other.shape[1] == R_sqrt.size
        out = R_sqrt * other  # Element-wise multiplication
    else:
        # For multi-orientation, each source has a 3x3 block.
        # Assert dimensions of R_sqrt: (n_src, 3, 3)
        assert R_sqrt.shape[1:3] == (3, 3)
        # other.shape[1] should be equal to (n_src*3)
        assert other.shape[1] == np.prod(R_sqrt.shape[:2])
        assert other.ndim == 2
        n_src = R_sqrt.shape[0]  # Number of sources
        n_chan = other.shape[0]  # Number of channels/sensors
        
        # Reshape and transpose to perform block multiplication
        out = (
            np.matmul(R_sqrt, other.reshape(n_chan, n_src, 3).transpose(1, 2, 0))
            .reshape(n_src * 3, n_chan)
            .T
        )
    return out

def compute_reginv2(sing, n_nzero, lambda2):
    """
    Compute the regularized inverse of singular values.
    
    This applies Tikhonov regularization in the SVD domain to handle small singular values.
    
    Parameters
    ----------
    sing : array-like, singular values from the SVD.
    n_nzero : int, number of non-zero singular values (typically number of sensors).
    lambda2 : float, regularization parameter.
    
    Returns
    -------
    reginv : array-like, the regularized inverses.
    """
    # Ensure the singular values are in floating point for precision.
    sing = np.array(sing, dtype=np.float64)
    reginv = np.zeros_like(sing)  # Initialize the output array
    # Consider only the first n_nzero singular values.
    sing = sing[:n_nzero]
    with np.errstate(invalid="ignore"):
        # Regularized inversion: sigma / (sigma^2 + lambda2)
        reginv[:n_nzero] = np.where(sing > 0, sing / (sing ** 2 + lambda2), 0)
    return reginv

def compute_orient_prior(G, n_orient, loose=0.9):
    """
    Compute an orientation prior for sources.
    
    The orientation prior weights help to scale the source estimates according
    to expected orientation variability (e.g., "loose" constraints for x and y directions).
    
    Parameters
    ----------
    G : ndarray, the lead-field matrix.
    n_orient : int, number of orientations per source.
    loose : float, scaling factor for certain orientations.
    
    Returns
    -------
    orient_prior : ndarray, shape (n_sources * n_orient,)
        The prior weights for each source orientation.
    """
    n_sources = G.shape[1]
    orient_prior = np.ones(n_sources, dtype=np.float64)  # Default is weight of 1 for all sources
    if n_orient == 1:
        return orient_prior  # No adjustment needed for single orientation
    # For multi-orientation (e.g., free orientation with three components),
    # the x and y orientations are scaled by the 'loose' factor.
    orient_prior[::3] *= loose  # Scale the first orientation (x)
    orient_prior[1::3] *= loose  # Scale the second orientation (y)
    # The third orientation (z) remains unchanged (multiplied by 1)
    return orient_prior

def safe_svd(A, full_matrices=False):
    """
    Safely compute the SVD of matrix A.
    
    Parameters
    ----------
    A : ndarray
        The matrix for which to compute the singular value decomposition.
    full_matrices : bool
        Flag determining if full or reduced SVD is computed.
    
    Returns
    -------
    U, S, Vh : ndarrays
        The left singular vectors, singular values, and right singular vectors.
    """
    return np.linalg.svd(A, full_matrices=full_matrices)

def compute_eloreta_kernel(L, *, lambda2, n_orient, whitener, loose=1.0, max_iter=20, logger=None):
    """
    Compute the eLORETA kernel and the posterior source covariance.
    
    This function carries out the main steps of the eLORETA estimation:
      1. Whiten the lead-field matrix L.
      2. Apply the orientation prior to the source covariance.
      3. Initialize and iteratively update the source covariance matrix R.
      4. Normalize R and compute the effective gain matrix.
      5. Perform an SVD on the effective gain matrix and regularize the singular values.
      6. Assemble the final inverse operator (kernel K).
    
    Parameters
    ----------
    L : ndarray, shape (n_chan, n_src*n_orient)
        The original lead-field matrix.
    lambda2 : float, regularization parameter to stabilize the inversion.
    n_orient : int, the number of orientations per source (1 for fixed orientation, 3 for free orientation).
    whitener : ndarray, the whitening matrix derived from the noise covariance.
    loose : float, parameter for the orientation prior (looseness of the constraints).
    max_iter : int, maximum number of iterations for the iterative fitting procedure.
    
    Returns
    -------
    K : ndarray, the eLORETA kernel (inverse operator) used to compute source estimates.
    Sigma : ndarray, the posterior source covariance matrix.
    """
    options = dict(eps=1e-6, max_iter=max_iter, force_equal=False)  # taken from mne
    eps, max_iter = options["eps"], options["max_iter"]
    force_equal = bool(options["force_equal"])  # None means False

    G = whitener @ L
    n_nzero = G.shape[0]

    # restore orientation prior
    source_std = np.ones(G.shape[1])

    orient_prior = compute_orient_prior(G, n_orient, loose=loose)
    source_std *= np.sqrt(orient_prior)

    G *= source_std

    # We do not multiply by the depth prior, as eLORETA should compensate for
    # depth bias.
    _, n_src = G.shape
    n_src //= n_orient

    assert n_orient in (1, 3)

    # src, sens, 3
    G_3 = get_G_3(G, n_orient)
    if n_orient != 1 and not force_equal:
        # Outer product
        R_prior = source_std.reshape(n_src, 1, 3) * source_std.reshape(n_src, 3, 1)
    else:
        R_prior = source_std ** 2

    # The following was adapted under BSD license by permission of Guido Nolte
    if force_equal or n_orient == 1:
        R_shape = (n_src * n_orient,)
        R = np.ones(R_shape)
    else:
        R_shape = (n_src, n_orient, n_orient)
        R = np.empty(R_shape)
        R[:] = np.eye(n_orient)[np.newaxis]
    R *= R_prior
    this_normalize_R = partial(
        normalize_R,
        n_nzero=n_nzero,
        force_equal=force_equal,
        n_src=n_src,
        n_orient=n_orient,
    )
    G_R_Gt = this_normalize_R(G, R, G_3)
    # extra = " (this make take a while)" if n_orient == 3 else ""
    for kk in range(max_iter):
        # 1. Compute inverse of the weights (stabilized) and C
        s, u = eigh(G_R_Gt)
        s = abs(s)
        sidx = np.argsort(s)[::-1][:n_nzero]
        s, u = s[sidx], u[:, sidx]
        with np.errstate(invalid="ignore"):
            s = np.where(s > 0, 1 / (s + lambda2), 0)
        N = np.dot(u * s, u.T)
        del s

        # Update the weights
        R_last = R.copy()
        if n_orient == 1:
            R[:] = 1.0 / np.sqrt((np.dot(N, G) * G).sum(0))
        else:
            M = np.matmul(np.matmul(G_3, N[np.newaxis]), G_3.swapaxes(-2, -1))
            if force_equal:
                _, s = sqrtm_sym(M, inv=True)
                R[:] = np.repeat(1.0 / np.mean(s, axis=-1), 3)
            else:
                R[:], _ = sqrtm_sym(M, inv=True)
        R *= R_prior  # reapply our prior, eLORETA undoes it
        G_R_Gt = this_normalize_R(G, R, G_3)

        # Check for weight convergence
        delta = np.linalg.norm(R.ravel() - R_last.ravel()) / np.linalg.norm(
            R_last.ravel()
        )
        if delta < eps:
            break
    else:
        logger.debug("eLORETA weight fitting did not converge (>= %s)" % eps)
    del G_R_Gt
    G /= source_std  # undo our biasing
    G_3 = get_G_3(G, n_orient)
    this_normalize_R(G, R, G_3)
    del G_3
    if n_orient == 1 or force_equal:
        R_sqrt = np.sqrt(R)
    else:
        R_sqrt = sqrtm_sym(R)[0]
    assert R_sqrt.shape == R_shape
    A = R_sqrt_mult(G, R_sqrt)
    eigen_fields, sing, eigen_leads = safe_svd(A, full_matrices=False)
    
    # Precompute regularization terms for K and Σ_X
    reginv_k = compute_reginv2(sing, n_nzero, lambda2)  # σ_i / (σ_i² + λ)
    reginv_s = sing * reginv_k  # σ_i² / (σ_i² + λ) = σ_i * (σ_i / (σ_i² + λ))
    
    # Compute K using existing terms
    eigen_leads = R_sqrt_mult(eigen_leads, R_sqrt).T
    trans = np.dot(eigen_fields.T, whitener)
    trans *= reginv_k[:, None]
    K = np.dot(eigen_leads, trans)
    
    # Compute Σ_X directly from V and reginv_s
    eigen_leads_t = eigen_leads.T
    eigen_leads_t *= reginv_s[:, None] # each row scaled by σ_i / (σ_i² + λ)
    Sigma = R - np.dot(eigen_leads, eigen_leads_t)
    
    return K, Sigma

def eloreta(L, y, noise_var,  n_orient=1, verbose=True, logger=None, **kwargs):
    """
    Compute the eLORETA solution for EEG/MEG inverse modeling.
    
    This is the main interface function that:
      - Preprocesses the lead-field and data,
      - Applies noise whitening,
      - Computes the eLORETA kernel,
      - And finally estimates the source activity.
    
    Parameters
    ----------
    L : ndarray, shape (n_chan, n_src*n_orient)
        The lead-field (forward) matrix mapping sources to sensors.
    y : ndarray, shape (n_chan, n_times) or (n_chan,)
        The sensor data (EEG/MEG recordings) to be inverted.

    n_orient : int
        Number of orientations per source (1 for fixed or 3 for free orientation).
    
    Returns
    -------
    x : ndarray
        The estimated source activations. The shape will be (n_src, n_times) for
        single orientation or (n_src, n_orient, n_times) for free orientations.
    Sigma : ndarray
        The posterior source covariance, characterizing the uncertainty in estimates.
    """        
    # TODO: check if this work for all noise types
    noise_cov = noise_var * np.eye(L.shape[0]) 
    
    # Create the whitening matrix from the noise covariance:
    # Typically computed as the inverse of the square root of the covariance.
    whitener = linalg.inv(linalg.sqrtm(noise_cov))
    
    # Whiten both the sensor data and the lead-field matrix.
    y = whitener @ y
    L = whitener @ L

    # Compute the eLORETA kernel and the posterior source covariance using the helper.
    # alpha is lambda2 = noise_var
    K, Sigma = compute_eloreta_kernel(
        L, 
        lambda2=1.0,
        n_orient=n_orient,
        whitener=whitener,
        logger=logger
    )
    
    # Compute the mean source estimates.
    x = K @ y # get the source time courses with simple dot product

    # If using free orientation sources (n_orient > 1), reshape the output.
    if n_orient > 1:
        x = x.reshape((-1, n_orient, x.shape[1]))

    active_indices = np.arange(Sigma.shape[0])  # All sources are active in eLORETA
    return {
        "posterior_mean": x,
        "posterior_cov": Sigma,
        "noise_var": noise_var,
        "active_indices": active_indices,
    }


# ==========================================
# BMN (with sLORETA normalization) Functions
# ==========================================

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

# sLORETA normalization
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

# BMN with fixed known noise variance
def BMN_opt(
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
    **kwargs,
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

    x_hat_normal, posterior_cov_normal, gamma = BMN_opt(
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
# BMN with noise learning API
# =============================================================================

# Convex-bounding update rule for common scalar noise variance parameter
def update_common_lambda_convex(
    y: np.ndarray,
    L: np.ndarray,
    posterior_mean: np.ndarray,
    C_inv: np.ndarray,
    eps: float = 1e-12,
) -> float:
    """
    Convex-bounding update rule for one common scalar noise variance in normalized scale.

    This function returns alpha_new, where
        alpha = lambda / ||Y||_F^2

    Formula
    -------
    alpha = sqrt( ( ||Y - L X||_F^2 / T ) / tr(C^{-1}) )
    """
    _, T = y.shape

    residual = y - L @ posterior_mean
    residual_term = (np.linalg.norm(residual, ord="fro") ** 2) / T
    denominator = np.trace(C_inv)

    alpha_new = np.sqrt(max(residual_term, 0.0) / max(denominator, eps))
    return max(float(alpha_new), eps)

# BMN optimization with optional adaptive noise learning
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

def BMN_joint(
    L: np.ndarray,
    y: np.ndarray,
    noise_var: Optional[float] = None,
    n_orient: int = 1,
    max_iter: int = 1000,
    tol: float = 1e-6,
    init_gamma: Optional[float] = None,
    init_lambda: Optional[float] = None,
    learn_noise: bool = True,
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


# ==================
# Main Solver Class
# ==================
class SourceEstimator(BaseEstimator, ClassifierMixin):
    def __init__(self, solver, solver_params=None, noise_var=None, n_orient=1, logger=None):
        """
        Initialize the SourceEstimator class.

        Parameters
        ----------
        solver : callable
            The inverse solver function (e.g., gamma_map_sflex, BMN).
        solver_params : dict, optional
            Parameters for the solver function.
        noise_var : float, optional
            Noise variance for the solver.
        logger : logging.Logger, optional
            Logger instance for logging messages.
        n_orient : int, optional
            Number of orientations for the sources. Default is 1 (for fixed
            orientation) or 3 (for free orientation).
        """
        # Follow sklearn convention: __init__ should *only* assign the passed
        # parameters to attributes without mutating them. Keep `solver_params`
        # as provided (None is allowed) so that clone can reconstruct the
        # estimator exactly. Downstream code should treat None as an empty
        # dict when invoking the solver.
        self.solver = solver
        self.solver_params = solver_params
        self.noise_var = noise_var
        self.logger = logger or logging.getLogger(__name__)
        self.n_orient = n_orient

    def _format_leadfield(self, L):
        """
        Ensure the leadfield matches the solver expectation of (n_sensors, n_sources * n_orient).

        Parameters
        ----------
        L : np.ndarray
            Leadfield array with shape (n_sensors, n_sources) for fixed-orientation
            or (n_sensors, n_sources, n_orient) for free-orientation setups.

        Returns
        -------
        np.ndarray
            A 2-D leadfield with shape (n_sensors, n_sources * n_orient).
        """
        if L.ndim == 2:
            return L
        if L.ndim == 3:
            n_sensors, n_sources, n_vec = L.shape
            if self.n_orient not in (None, n_vec):
                self.logger.debug(
                    "Updating n_orient from %s to %s based on leadfield shape. Setting n_orient to %s.",
                    self.n_orient,
                    n_vec,
                )
                self.n_orient = n_vec
            if n_vec not in (1, 3):
                self.logger.warning(
                    "Leadfield last dimension is %s; expected orientation components "
                    "of size 1 or 3.",
                    n_vec,
                )
            return L.reshape(n_sensors, n_sources * n_vec)
        raise ValueError(f"Leadfield must be 2-D or 3-D, got shape {L.shape}")

    def fit(self, L, y):
        """
        Fit the inverse solver to the data.

        Parameters
        ----------
        L : np.ndarray
            Leadfield matrix of shape (n_sensors, n_sources) for fixed
            orientation or (n_sensors, n_sources, n_orient) for free orientation.
        y : np.ndarray
            Observed EEG/MEG signals of shape (n_sensors, n_times).

        Returns
        -------
        self
            The fitted estimator.
        """
        self.logger.debug("Fitting the solver...")
        self.L_ = self._format_leadfield(L)
        self.y_ = y
        
        return self

    def _get_coef(self, y):
        """
        Internal method to compute the source estimates.

        Parameters
        ----------
        y : np.ndarray
            Observed EEG/MEG signals of shape (n_sensors, n_times).

        Returns
        -------
        x_hat : np.ndarray
            Estimated source activity of shape (n_sources, n_times).
        - active_indices (np.ndarray): Indices of active sources.
        - posterior_cov (np.ndarray): Posterior covariance matrix of estimated sources.
        """        
        # Apply the solver
        if y is None:
            if not hasattr(self, "y_"):
                raise ValueError("No data available to compute source estimates. Fit the estimator or pass y.")
            y = self.y_
        solver_name = getattr(self.solver, "__name__", self.solver.__class__.__name__)
        self.logger.debug(f"Estimating sources using {solver_name}...")
        
        solver_kwargs = dict(self.solver_params or {})
        solver_kwargs.update(
            {
                "L": self.L_,
                "y": y,
                "n_orient": self.n_orient,
                "logger": self.logger,
            }
        )

        try:
            # Try passing noise_var if the solver accepts it
            return self.solver(noise_var=self.noise_var, **solver_kwargs)
        except TypeError as err:
            if "noise_var" not in str(err):
                raise # re-raise unexpected TypeErrors
            # fallback for solvers that do not accept noise_var argument (e.g. joint learning with gamma_lambda_map_sflex())
            return self.solver(**solver_kwargs)
    
    def predict(self, y=None):
        if y is None:
            if not hasattr(self, "y_"):
                raise ValueError("Estimator has not been fitted and no data was provided to predict().")
            y = self.y_
        return self._get_coef(y)

# =================
# Gamma-MAP with Joint Learning
# =================
def _as_2d_y(y: np.ndarray) -> np.ndarray:
    y = np.asarray(y, dtype=float)
    if y.ndim == 1:
        y = y[:, None]
    if y.ndim != 2:
        raise ValueError("y must have shape (M,T) or (M,).")
    return y

def _validate_inverse_inputs(
    L: np.ndarray,
    y: np.ndarray,
    n_orient: int,
) -> Tuple[np.ndarray, np.ndarray]:
    L = np.asarray(L, dtype=float)
    y = _as_2d_y(y)

    if L.ndim != 2:
        raise ValueError("L must be 2D.")
    if L.shape[0] != y.shape[0]:
        raise ValueError("L and y must have the same number of sensor rows.")
    if n_orient not in (1, 2, 3):
        raise ValueError("n_orient must be 1, 2, or 3.")
    if L.shape[1] % n_orient != 0:
        raise ValueError(
            f"For n_orient={n_orient}, L must have k*N columns with k={n_orient}."
        )
    return L, y

def _expand_grouped_parameter(
    value,
    n_coeff: int,
    group_size: int,
    name: str,
) -> np.ndarray:
    """
    Accepts:
      - None
      - scalar
      - length n_coeff
      - length n_groups
    and returns length n_coeff.
    """
    if n_coeff % group_size != 0:
        raise ValueError("n_coeff must be divisible by group_size.")

    n_groups = n_coeff // group_size

    if value is None:
        return np.ones(n_coeff, dtype=float)

    if np.isscalar(value):
        return np.full(n_coeff, float(value), dtype=float)

    value = np.asarray(value, dtype=float).ravel()

    if value.size == n_coeff:
        return value.copy()

    if value.size == n_groups:
        return np.repeat(value, group_size).astype(float)

    if isinstance(value, tuple) and len(value) == 2:
        return np.linspace(value[0], value[1], num=n_coeff).astype(float)

    raise ValueError(
        f"{name} must be None, scalar, length {n_coeff}, or length {n_groups}."
    )

def _build_sflex_operator(B, n_orient: int) -> csr_matrix:
    """
    Build coefficient-space sFLEX operator.

    For flattened coefficient ordering
        [source1 comp1..k, source2 comp1..k, ...],
    the correct operator is:
        B ⊗ I_k
    """
    if issparse(B):
        B = B.tocsr()
    else:
        B = csr_matrix(np.asarray(B, dtype=float))

    if B.shape[0] != B.shape[1]:
        raise ValueError("B must be square.")

    if n_orient == 1:
        return B

    Ik = eye(n_orient, format="csr")
    return kron(B, Ik, format="csr")

def _lambda_opt(
    M: np.ndarray,
    G_active: np.ndarray,
    x_active_norm: np.ndarray,
    posterior_cov_active_norm: np.ndarray,
    current_lambda_norm: np.ndarray,
    CMinv: np.ndarray,
    update_mode_noise: int,
) -> np.ndarray:
    """
    Update diagonal lambda in NORMALIZED scale.

    update_mode_noise
    -----------------
    1 : EM-style variance update
    2 : Convex-bounding style update
    """
    M = np.asarray(M, dtype=float)
    G_active = np.asarray(G_active, dtype=float)
    x_active_norm = np.asarray(x_active_norm, dtype=float)
    posterior_cov_active_norm = np.asarray(posterior_cov_active_norm, dtype=float)
    current_lambda_norm = np.asarray(current_lambda_norm, dtype=float)
    CMinv = np.asarray(CMinv, dtype=float)

    n_sensors, T = M.shape
    eps = 1e-16
    lam_new = np.zeros(n_sensors, dtype=float)

    if update_mode_noise == 1:
        for m in range(n_sensors):
            residual = M[m, :] - (G_active[m, :] @ x_active_norm)
            residual_term = float(np.mean(residual**2))
            g_m = G_active[m, :]
            cov_term = float(g_m @ posterior_cov_active_norm @ g_m)
            lam_new[m] = residual_term + cov_term

    elif update_mode_noise == 2:
        for m in range(n_sensors):
            residual = M[m, :] - (G_active[m, :] @ x_active_norm)
            numerator = float(np.mean(residual**2))
            denom = float(CMinv[m, m])

            if denom > eps:
                lam_new[m] = np.sqrt(max(numerator, 0.0) / denom)
            else:
                lam_new[m] = current_lambda_norm[m]
    else:
        raise ValueError("update_mode_noise must be 1 or 2.")

    return np.maximum(lam_new, eps)

def _gamma_lambda_map_opt(
    M: np.ndarray,
    G: np.ndarray,
    *,
    maxit: int = 300,
    tol: float = 1e-6,
    update_mode: int = 2,
    group_size: int = 1,
    init_gamma=None,
    init_lambda=None,
    learn_lambda: bool = True,
    update_mode_noise: int = 2,
    lambda_damping: float = 1.0,
    track_history: bool = True,
    verbose: bool = False,
    logger=None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[str, list]]:
    """
    Internal optimizer for grouped gamma MAP with diagonal adaptive lambda.

    Important conventions
    ---------------------
    - grouped gamma updates are preserved through group_size
    - init_lambda=None means ones in the INTERNAL normalized scale
    - user-supplied init_lambda is interpreted in ORIGINAL sensor-variance units
    """
    if logger is None:
        logger = logging.getLogger(__name__)

    G = np.asarray(G, dtype=float).copy()
    M = _as_2d_y(M).copy()

    n_coeff = G.shape[1]
    n_sensors, n_times = M.shape
    eps = np.finfo(float).eps

    if n_coeff % group_size != 0:
        raise ValueError("Number of coefficients must be divisible by group_size.")

    # ------------------------------------------------------------
    # Normalize M and G for numerical stability
    # ------------------------------------------------------------
    M_norm_c = float(np.linalg.norm(M @ M.T, ord="fro"))
    if M_norm_c <= 0:
        raise ValueError("Degenerate M.")

    G_norm_c = float(np.linalg.norm(G, ord=np.inf))
    if G_norm_c <= 0:
        raise ValueError("Degenerate G.")

    M /= (np.sqrt(M_norm_c) + eps)
    G /= (G_norm_c + eps)

    # ------------------------------------------------------------
    # Init gamma
    # ------------------------------------------------------------
    gammas_full_old = _expand_grouped_parameter(
        init_gamma,
        n_coeff=n_coeff,
        group_size=group_size,
        name="init_gamma",
    ).astype(float)
    gammas_full_old = np.maximum(gammas_full_old, 0.0)

    # ------------------------------------------------------------
    # Init lambda
    # ------------------------------------------------------------
    # FIX:
    #   init_lambda=None -> ones in INTERNAL NORMALIZED scale
    #   user-supplied init_lambda -> ORIGINAL scale, then normalized
    if init_lambda is None:
        if learn_lambda:
            current_lambda = np.ones(n_sensors, dtype=float)
        else:
            raise ValueError("learn_lambda=False requires init_lambda.")
    else:
        if np.isscalar(init_lambda):
            lambda_orig = np.full(n_sensors, float(init_lambda), dtype=float)
        else:
            lambda_orig = np.asarray(init_lambda, dtype=float).ravel()
            if lambda_orig.size != n_sensors:
                raise ValueError(
                    f"init_lambda must be scalar or length {n_sensors}."
                )
        current_lambda = np.maximum(lambda_orig, eps) / (M_norm_c + eps)

    denom_fun = np.sqrt if update_mode == 2 else (lambda x: x)

    hist: Dict[str, list] = {}
    if track_history:
        hist["n_active_hist"] = []
        hist["err_gamma_hist"] = []
        hist["lambda_mean_hist"] = []

    active_indices = np.arange(n_coeff, dtype=int)
    gammas_active_new = None
    posterior_cov_active_norm = None
    A = None
    G_CMinvG = None

    last_size = -1

    for itno in range(int(maxit)):
        gammas_active = gammas_full_old[active_indices]
        gammas_active = np.nan_to_num(gammas_active, nan=0.0, posinf=0.0, neginf=0.0)

        keep = np.abs(gammas_active) > eps
        active_indices = active_indices[keep]
        gammas_active = gammas_active[keep]

        if active_indices.size == 0:
            break

        if active_indices.size % group_size != 0:
            raise RuntimeError(
                "Active coefficient count is not divisible by group_size. "
                "Grouped coefficients must remain together."
            )

        G_active = G[:, active_indices]

        # CM = G diag(gamma) G^T + diag(lambda)
        CM = (G_active * gammas_active[None, :]) @ G_active.T
        np.fill_diagonal(CM, CM.diagonal() + current_lambda)

        try:
            U, S, _ = linalg.svd(CM, full_matrices=False)
            CMinv = (U / (S[None, :] + eps)) @ U.T
        except linalg.LinAlgError:
            CMinv = linalg.pinv(CM)

        CMinvG = CMinv @ G_active
        A = CMinvG.T @ M  # (K_active, T)

        # --------------------------------------------------------
        # Gamma update
        # --------------------------------------------------------
        if update_mode == 1:
            numer = gammas_active**2 * np.mean((A * A.conj()).real, axis=1)
            denom = gammas_active * np.sum(G_active * CMinvG, axis=0)

        elif update_mode == 2:
            numer = gammas_active * np.sqrt(np.mean((A * A.conj()).real, axis=1))
            denom = np.sum(G_active * CMinvG, axis=0)

        elif update_mode == 3:
            denom = None
            numer = gammas_active**2 * np.mean((A * A.conj()).real, axis=1) + gammas_active * (
                1.0 - gammas_active * np.sum(G_active * CMinvG, axis=0)
            )
        else:
            raise ValueError("Invalid update_mode. Use 1, 2, or 3.")

        if group_size == 1:
            if denom is None:
                gammas_active_new = numer
            else:
                gammas_active_new = numer / np.maximum(denom_fun(denom), eps)
        else:
            numer_comb = np.sum(numer.reshape(-1, group_size), axis=1)

            if denom is None:
                gammas_comb = numer_comb
            else:
                denom_comb = np.sum(denom.reshape(-1, group_size), axis=1)
                gammas_comb = numer_comb / np.maximum(denom_fun(denom_comb), eps)

            gammas_active_new = np.repeat(gammas_comb / group_size, group_size)

        gammas_active_new = np.maximum(gammas_active_new, 0.0)

        # --------------------------------------------------------
        # Posterior covariance in normalized coefficient space
        # --------------------------------------------------------
        G_CMinvG = G_active.T @ CMinvG
        posterior_cov_active_norm = (
            np.diag(gammas_active_new)
            - gammas_active_new[:, None] * G_CMinvG * gammas_active_new[None, :]
        )
        posterior_cov_active_norm = _symmetrize(posterior_cov_active_norm)

        # --------------------------------------------------------
        # Lambda update
        # --------------------------------------------------------
        if learn_lambda:
            x_active_norm = gammas_active_new[:, None] * A
            lam_new = _lambda_opt(
                M=M,
                G_active=G_active,
                x_active_norm=x_active_norm,
                posterior_cov_active_norm=posterior_cov_active_norm,
                current_lambda_norm=current_lambda,
                CMinv=CMinv,
                update_mode_noise=update_mode_noise,
            )
            d = float(np.clip(lambda_damping, 0.0, 1.0))
            current_lambda = (1.0 - d) * current_lambda + d * lam_new

        gammas_full = np.zeros(n_coeff, dtype=float)
        gammas_full[active_indices] = gammas_active_new

        err = np.sum(np.abs(gammas_full - gammas_full_old)) / (
            np.sum(np.abs(gammas_full_old)) + eps
        )
        gammas_full_old = gammas_full

        if track_history:
            hist["n_active_hist"].append(int(active_indices.size))
            hist["err_gamma_hist"].append(float(err))
            hist["lambda_mean_hist"].append(float(np.mean(current_lambda) * M_norm_c))

        breaking = (err < tol) or (active_indices.size == 0)

        if verbose and ((active_indices.size != last_size) or breaking):
            logger.info(
                f"it={itno:3d} active={active_indices.size:4d} "
                f"err_gamma={err:.3e} "
                f"lambda_mean(orig)={np.mean(current_lambda) * M_norm_c:.3e}"
            )
            last_size = active_indices.size

        if breaking:
            break

    # ------------------------------------------------------------
    # Empty solution
    # ------------------------------------------------------------
    if gammas_active_new is None or active_indices.size == 0:
        x_active = np.zeros((0, n_times), dtype=float)
        cov_out = np.zeros((0, 0), dtype=float)
        gammas_full = np.zeros(n_coeff, dtype=float)
        lambda_final = current_lambda * M_norm_c
        return x_active, active_indices, cov_out, gammas_full, lambda_final, hist

    # ------------------------------------------------------------
    # Undo normalization back to original coefficient scale
    # ------------------------------------------------------------
    n_const = np.sqrt(M_norm_c) / (G_norm_c + eps)
    x_active = n_const * gammas_active_new[:, None] * A

    cov_out = (
        np.diag(gammas_active_new)
        - gammas_active_new[:, None] * G_CMinvG * gammas_active_new[None, :]
    )
    cov_out = (n_const**2) * cov_out
    cov_out = _symmetrize(cov_out)

    lambda_final = current_lambda * M_norm_c

    return x_active, active_indices, cov_out, gammas_full_old, lambda_final, hist

def gamma_lambda_map(
    L: np.ndarray,
    y: np.ndarray,
    n_orient: int = 1,
    init_gamma=None,
    init_lambda=None,
    max_iter: int = 300,
    tol: float = 1e-6,
    update_mode: int = 2,
    learn_lambda: bool = True,
    update_mode_noise: int = 2,
    lambda_damping: float = 1.0,
    track_history: bool = True,
    verbose: bool = False,
    logger=None,
) -> Dict[str, Any]:
    """
    Grouped gamma-lambda MAP with diagonal adaptive lambda.

    Supports
    --------
    n_orient = 1 : fixed
    n_orient = 2 : reduced free MEG
    n_orient = 3 : free EEG

    Returns
    -------
    dict with keys:
      posterior_mean            : (N,T) if n_orient=1 else (kN,T)
      posterior_mean_reshaped   : (N,k,T) for k>1
      posterior_cov             : full coefficient covariance, shape (kN,kN)
      posterior_cov_active      : active-only covariance
      active_indices            : active coefficient indices
      gammas_full               : length kN
      gamma                     : mean(gammas_full)
      lambdas                   : diagonal lambda vector, original scale
      lambda_mean               : mean diagonal lambda
    """
    if logger is None:
        logger = logging.getLogger(__name__)

    L, y = _validate_inverse_inputs(L=L, y=y, n_orient=n_orient)

    n_coeff = L.shape[1]
    n_sources = n_coeff // n_orient

    x_active, active_idx, cov_active, gammas_full, lambdas, hist = _gamma_lambda_map_opt(
        M=y,
        G=L,
        maxit=max_iter,
        tol=tol,
        update_mode=update_mode,
        group_size=n_orient,
        init_gamma=init_gamma,
        init_lambda=init_lambda,
        learn_lambda=learn_lambda,
        update_mode_noise=update_mode_noise,
        lambda_damping=lambda_damping,
        track_history=track_history,
        verbose=verbose,
        logger=logger,
    )

    x_hat = np.zeros((n_coeff, y.shape[1]), dtype=float)
    if active_idx.size > 0:
        x_hat[active_idx] = x_active

    posterior_cov = np.zeros((n_coeff, n_coeff), dtype=float)
    if active_idx.size > 0:
        posterior_cov[np.ix_(active_idx, active_idx)] = cov_active
    posterior_cov = _symmetrize(posterior_cov)

    out = {
        "posterior_mean": x_hat if n_orient > 1 else x_hat.reshape(n_sources, y.shape[1]),
        "posterior_cov": posterior_cov,
        "posterior_cov_active": cov_active,
        "active_indices": active_idx,
        "gamma": float(np.mean(gammas_full)) if gammas_full.size else 0.0,
        "gammas_full": gammas_full,
        "lambdas": np.asarray(lambdas, dtype=float),
        "lambda_mean": float(np.mean(lambdas)) if lambdas.size else 0.0,
        "noise_var": float(np.mean(lambdas)) if lambdas.size else 0.0,  # compatibility alias
        "coefficient_indices": np.arange(n_coeff),
        "source_indices": np.arange(n_sources),
    }

    if n_orient > 1:
        out["posterior_mean_reshaped"] = x_hat.reshape(n_sources, n_orient, y.shape[1])

    if track_history:
        out.update(hist)

    return out

def gamma_lambda_map_sflex(
    L: np.ndarray,
    y: np.ndarray,
    n_orient: int = 1,
    init_gamma=None,
    init_lambda=None,
    learn_lambda: bool = True,
    update_mode_noise: int = 2,
    lambda_damping: float = 1.0,
    max_iter: int = 300,
    tol: float = 1e-6,
    update_mode: int = 2,
    track_history: bool = True,
    sigma: float = 0.01,
    threshold_factor: float = 3.0,
    normalize: Optional[str] = "sym",
    eps: float = 1e-12,
    verbose: bool = False,
    logger=None,
    **kwargs,
) -> Dict[str, Any]:
    """
    Unified sFLEX + gamma-lambda MAP.

    Source model
    ------------
    x = (B ⊗ I_k) c

    Supports
    --------
    n_orient = 1 : fixed
    n_orient = 2 : reduced free MEG
    n_orient = 3 : free EEG

    Returns
    -------
    dict with posterior quantities in SOURCE space, plus coefficient-space
    auxiliaries for debugging / analysis.
    """
    if logger is None:
        logger = logging.getLogger(__name__)

    L, y = _validate_inverse_inputs(L=L, y=y, n_orient=n_orient)

    n_coeff = L.shape[1]
    n_sources = n_coeff // n_orient
    T = y.shape[1]

    B = compute_B(
        sigma=sigma,
        threshold_factor=threshold_factor,
        normalize=normalize,
        eps=eps,
        src_coords=kwargs.get("src_coords"),
    )
    
    if issparse(B):
        B = B.tocsr()
    else:
        B = csr_matrix(np.asarray(B, dtype=float))

    if B.shape != (n_sources, n_sources):
        raise ValueError(
            f"B must have shape ({n_sources},{n_sources}); got {B.shape}."
        )

    B_op = _build_sflex_operator(B, n_orient=n_orient)  # (kN, kN)
    G = L @ B_op

    res_coeff = gamma_lambda_map(
        L=G,
        y=y,
        n_orient=n_orient,
        init_gamma=init_gamma,
        init_lambda=init_lambda,
        max_iter=max_iter,
        tol=tol,
        update_mode=update_mode,
        learn_lambda=learn_lambda,
        update_mode_noise=update_mode_noise,
        lambda_damping=lambda_damping,
        track_history=track_history,
        verbose=verbose,
        logger=logger,
    )

    # coefficient posterior mean c_hat
    c_hat_flat = np.asarray(res_coeff["posterior_mean"], dtype=float)
    if n_orient == 1:
        c_hat_flat = c_hat_flat.reshape(n_sources, T)
    else:
        c_hat_flat = c_hat_flat.reshape(n_coeff, T)

    # map mean to source space: x = (B ⊗ I_k) c
    x_hat_flat = B_op @ c_hat_flat
    x_hat_flat = np.asarray(x_hat_flat, dtype=float)

    # map active covariance to source space
    active = np.asarray(res_coeff["active_indices"], dtype=int)
    Sigma_c_active = np.asarray(res_coeff["posterior_cov_active"], dtype=float)

    if active.size > 0:
        B_active = B_op[:, active]
        B_active_dense = B_active.toarray()
        posterior_cov_x = B_active_dense @ Sigma_c_active @ B_active_dense.T
        posterior_cov_x = _symmetrize(np.asarray(posterior_cov_x, dtype=float))
    else:
        posterior_cov_x = np.zeros((n_coeff, n_coeff), dtype=float)

    out = {
        "posterior_mean": x_hat_flat if n_orient > 1 else x_hat_flat.reshape(n_sources, T),
        "posterior_cov": posterior_cov_x,
        "posterior_cov_active": posterior_cov_x[np.ix_(active, active)] if active.size > 0 else np.zeros((0, 0)),
        "active_indices": active,
        "gamma": float(res_coeff["gamma"]),
        "gammas_full": np.asarray(res_coeff["gammas_full"], dtype=float),
        "lambdas": np.asarray(res_coeff["lambdas"], dtype=float),
        "lambda_mean": float(res_coeff["lambda_mean"]),
        "noise_var": float(res_coeff["lambda_mean"]),  # compatibility alias
        # coefficient-space extras
        "posterior_mean_coeff": c_hat_flat if n_orient > 1 else c_hat_flat.reshape(n_sources, T),
        "posterior_cov_coeff": np.asarray(res_coeff["posterior_cov"], dtype=float),
        "posterior_cov_active_coeff": np.asarray(res_coeff["posterior_cov_active"], dtype=float),
        "B_operator": B_op,
        "coefficient_indices": np.arange(n_coeff),
        "source_indices": np.arange(n_sources),
    }

    if n_orient > 1:
        out["posterior_mean_reshaped"] = x_hat_flat.reshape(n_sources, n_orient, T)
        out["posterior_mean_coeff_reshaped"] = c_hat_flat.reshape(n_sources, n_orient, T)

    if track_history:
        for key in ["n_active_hist", "err_gamma_hist", "lambda_mean_hist"]:
            if key in res_coeff:
                out[key] = res_coeff[key]

    return out
