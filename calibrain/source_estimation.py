import logging
import numpy as np
from matplotlib import cm

from sklearn.model_selection import GridSearchCV, check_cv
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin, clone
from scipy import linalg
from scipy.spatial.distance import pdist, squareform
from scipy.sparse import coo_matrix, kron, identity
from scipy.stats import chi2, norm
from functools import partial
import mne
from mne.utils import sqrtm_sym, eigh
from mne.io.constants import FIFF
from numpy.linalg import inv
from scipy.linalg import sqrtm
from calibrain.utils import get_data_path
from scipy.sparse import block_diag
from typing import Optional, Dict, Any, Tuple

# ===================
# GAMMA-MAP Functions
# ===================

def gamma_map(
    L,
    y,
    noise_var,
    n_orient=1,
    max_iter=1000,
    tol=1e-15,
    update_mode=2,
    init_gamma=None,
    verbose=True,
    logger=None,
    **kwargs
):
    # # sigma_squared: noise variance = diagonal of the covariance matrix, where all diagonal elements are equal.
    # if noise_type == "oracle":
    #     noise_cov = noise_var * np.eye(L.shape[0])
    
    # TODO: check whether we still need this
    if init_gamma is None:
        init_gamma = np.ones(L.shape[1], dtype=np.float64)
    elif isinstance(init_gamma, (float, np.float64, int, np.int64)):
        init_gamma = np.full((L.shape[1],), init_gamma, dtype=np.float64)
    elif len(init_gamma) == 2 and isinstance(init_gamma, tuple):
        init_gamma = np.linspace(init_gamma[0], init_gamma[1], num=L.shape[1])
    else:
        raise ValueError("init_gamma should be a float, a tuple of two floats, or a list of floats.")

    noise_cov = noise_var * np.eye(L.shape[0]) 
    
    # Create the whitening matrix from the noise covariance:
    # Typically computed as the inverse of the square root of the covariance.
    whitener = linalg.inv(linalg.sqrtm(noise_cov))
    
    # Whiten both the sensor data and the lead-field matrix.
    y = whitener @ y
    L = whitener @ L
    
    x_hat_, active_indices, posterior_cov, gammas_full = _gamma_map_opt(
        y,
        L,
        sigma_squared=1.0,
        tol=tol,
        maxit=max_iter,
        init_gamma=init_gamma,
        update_mode=update_mode,
        group_size=n_orient,
        verbose=verbose,
        logger=logger,
    )
    x_hat = np.zeros((L.shape[1], y.shape[1]))
    x_hat[active_indices] = x_hat_

    if n_orient > 1:
        x_hat = x_hat.reshape((-1, n_orient, x_hat.shape[1]))
        
    
    # take the norm of the vector gammas_full
    gammas_norm = np.linalg.norm(gammas_full)

    return {
        "posterior_mean": x_hat,
        "posterior_cov": posterior_cov,
        "noise_var": noise_var,
        "active_indices": active_indices,
        "gamma": gammas_norm,
    }

def _gamma_map_opt(
    M,
    G,
    sigma_squared,
    maxit=10000,
    tol=1e-6,
    update_mode=2,
    group_size=1,
    init_gamma=None,
    verbose=None,
    logger=None,
):
    """Hierarchical Bayes (Gamma-MAP).

    Parameters
    ----------
    M : array, shape=(n_sensors, n_times)
        Observation.
    G : array, shape=(n_sensors, n_sources)
        Forward operator.
    sigma_squared : float
        Regularization parameter (noise variance).
    maxit : int
        Maximum number of iterations.
    tol : float
        Tolerance parameter for convergence.
    group_size : int
        Number of consecutive sources which use the same gamma.
    update_mode : int
        Update mode, 1: MacKay update (default), 3: Modified MacKay update.
    init_gamma : array, shape=(n_sources,)
        Initial values for posterior variances (init_gamma). If None, a
        variance of 1.0 is used.
    %(verbose)s

    Returns
    -------
    X : array, shape=(n_active, n_times)
        Estimated source time courses.
    active_indices : array, shape=(n_active,)
        Indices of active sources.
    posterior_cov: array, shape=(n_active, n_active)
        Posterior coveriance matrix of estimated active sources
    """
    G = G.copy()
    M = M.copy()

    n_sources = G.shape[1]
    n_sensors, n_times = M.shape
    
    eps = np.finfo(float).eps

    # apply normalization so the numerical values are sane
    M_normalize_constant = np.linalg.norm(np.dot(M, M.T), ord="fro")
    M /= np.sqrt(M_normalize_constant)
    sigma_squared /= M_normalize_constant
    G_normalize_constant = np.linalg.norm(G, ord=np.inf)
    G /= G_normalize_constant

    if n_sources % group_size != 0:
        raise ValueError(
            "Number of sources has to be evenly dividable by the " "group size"
        )

    n_active = n_sources
    active_indices = np.arange(n_sources)

    gammas_full_old = init_gamma.copy()

    if update_mode == 2:
        denom_fun = np.sqrt
    else:
        # do nothing
        def denom_fun(x):
            return x

    last_size = -1
    for itno in range(maxit):
        init_gamma[np.isnan(init_gamma)] = 0.0

        gidx = np.abs(init_gamma) > eps
        active_indices = active_indices[gidx]
        init_gamma = init_gamma[gidx]

        # update only active init_gamma (once set to zero it stays at zero)
        if n_active > len(active_indices):
            n_active = active_indices.size
            G = G[:, gidx]

        CM = np.dot(G * init_gamma[np.newaxis, :], G.T)
        CM.flat[:: n_sensors + 1] += sigma_squared
        # Invert CM keeping symmetry
        U, S, _ = linalg.svd(CM, full_matrices=False)
        S = S[np.newaxis, :]
        del CM
        CMinv = np.dot(U / (S + eps), U.T)
        CMinvG = np.dot(CMinv, G)
        A = np.dot(CMinvG.T, M)  # mult. w. Diag(gamma) in gamma update
        # G_CMinvG = G.T @ CMinvG

        if update_mode == 1:
            # MacKay fixed point update (10) in [1]
            numer = init_gamma ** 2 * np.mean((A * A.conj()).real, axis=1)
            denom = init_gamma * np.sum(G * CMinvG, axis=0)
        elif update_mode == 2:
            # modified MacKay fixed point update (11) in [1]
            numer = init_gamma * np.sqrt(np.mean((A * A.conj()).real, axis=1))
            denom = np.sum(G * CMinvG, axis=0)  # sqrt is applied below
        elif update_mode == 3:
            # Expectation Maximization (EM) update
            denom = None
            numer = init_gamma ** 2 * np.mean((A * A.conj()).real, axis=1) + init_gamma * (
                1 - init_gamma * np.sum(G * CMinvG, axis=0)
            )
        else:
            raise ValueError("Invalid value for update_mode")

        if group_size == 1:
            if denom is None:
                init_gamma = numer
            else:
                init_gamma = numer / np.maximum(denom_fun(denom), np.finfo("float").eps)
        else:
            numer_comb = np.sum(numer.reshape(-1, group_size), axis=1)
            if denom is None:
                gammas_comb = numer_comb
            else:
                denom_comb = np.sum(denom.reshape(-1, group_size), axis=1)
                gammas_comb = numer_comb / denom_fun(denom_comb)

            init_gamma = np.repeat(gammas_comb / group_size, group_size)

        # compute convergence criterion
        gammas_full = np.zeros(n_sources, dtype=np.float64)
        gammas_full[active_indices] = init_gamma

        err = np.sum(np.abs(gammas_full - gammas_full_old)) / np.sum(
            np.abs(gammas_full_old)
        )

        gammas_full_old = gammas_full

        breaking = err < tol or n_active == 0
        if len(init_gamma) != last_size or breaking:
            logger.debug(
                "Iteration: %d\t active set size: %d\t convergence: "
                "%0.3e" % (itno, len(init_gamma), err)
            )
            last_size = len(init_gamma)

        if breaking:
            break

    if itno < maxit - 1:
        logger.debug(
            "Iteration: %d\t active set size: %d\t convergence: "
            "%0.3e" % (itno, len(init_gamma), err)
        )
        logger.info("\nConvergence reached !\n")
    else:
        logger.debug(
            "Iteration: %d\t active set size: %d\t convergence: "
            "%0.3e" % (itno, len(init_gamma), err)
        )
        logger.debug("\nConvergence NOT reached !\n")

    # undo normalization and compute final posterior mean and posterior covariance
    n_const = np.sqrt(M_normalize_constant) / G_normalize_constant
    x_active = n_const * init_gamma[:, None] * A



    # Compute the posterior convariance matrix as in eq. (2.10) in Hashemi, Ali. "Advances in hierarchical Bayesian learning with applications to neuroimaging." (2023).
    # pos_cov =  np.diag(init_gamma) - init_gamma[:, np.newaxis] * G_CMinvG * init_gamma
    posterior_cov = np.diag(init_gamma) - init_gamma[:, np.newaxis] * G.T @ CMinv @ G * init_gamma 

    # Undo normalization for posterior covariance (similar to x_orig = n_const * x_norm)
    posterior_cov = (n_const ** 2) * posterior_cov
    
    # A similar approach can be implemented (as Large_gamma is interpreted as a diagonal matrix with small_gammas:
    # posterior_cov = np.diag(init_gamma) - np.diag(init_gamma) @ G.T @ CMinv @ G @ np.diag(init_gamma)
    
    return x_active, active_indices, posterior_cov, gammas_full


# ==================
# sFlex Functions
# ==================

def compute_B(src_coords, sigma, threshold_factor=3.0):
    """
    Compute sFLEX basis matrix B using truncated RBF weights.

    Parameters
    ----------
    src_coords : ndarray, shape (N,3)
        Source coordinates.
    sigma : float
        RBF width.
    threshold_factor : float
        Truncation radius multiplier.

    Returns
    -------
    B : scipy.sparse.csr_matrix, shape (N,N)
        Sparse basis matrix.
    """
    src_coords = np.asarray(src_coords, float)
    N, D = src_coords.shape
    if D != 3:
        raise ValueError("src_coords must have shape (N,3).")

    dist2 = squareform(pdist(src_coords, 'sqeuclidean'))
    r2 = (threshold_factor * sigma) ** 2
    mask = dist2 <= r2
    rows, cols = np.nonzero(mask)

    pref = (2.0 * np.pi * sigma**2) ** (-1.5)
    weights = pref * np.exp(-dist2[rows, cols] / (2.0 * sigma**2))

    B = coo_matrix((weights, (rows, cols)), shape=(N, N))
    B = (B + B.T) * 0.5
    return B.tocsr()



def sflex_gamma_map(L, y, noise_var, fwd_path, sigma=0.001, n_orient=1, max_iter=1000, tol=1e-15, update_mode=2, init_gamma=None, verbose=True, logger=None, threshold_factor=3.0, **kwargs):
    """
    Unified s-FLEX + γ-MAP implementation for both fixed and free orientation cases.
    
    Parameters
    ----------
    L : ndarray
        Lead field matrix.
    y : ndarray
        Sensor measurements.
    sigma : float
        Gaussian basis width parameter.
    noise_var : float
        Noise variance.
    fwd_path : str or Path
        Path to the forward solution file.
    n_orient : int
        Number of orientations (1 for fixed, 3 for free).
    max_iter : int
        Maximum number of iterations.
    tol : float
        Convergence tolerance.
    update_mode : int
        Gamma update mode.
    init_gamma : ndarray or None
        Initial gamma values.
    verbose : bool
        Whether to print progress.
    logger : object
        Logger object for progress reporting.
    threshold_factor : float
        Threshold factor for basis sparsity.
        
    Returns
    -------
    x_hat : ndarray
        Estimated source activity.
    active_indices : ndarray
        Indices of active sources.
    posterior_cov : ndarray
        Posterior covariance of active sources. In gamma_map it is the active coefficients’ covariance,
        whereas in sflex_gamma_map it is the source-space covariance obtained by mapping it with B.
    """
    fwd_path = f"{fwd_path}-fwd.fif"
    fwd = mne.read_forward_solution(fwd_path, verbose="error")

    if n_orient == 2 and fwd['source_ori'] == FIFF.FIFFV_MNE_FREE_ORI:
        fwd = mne.convert_forward_solution(fwd, force_fixed=True)
        
    # src_coords = fwd['src'][0]['rr']  # (N, 3) source locations in meters
    src_coords = fwd['source_rr']  # (N, 3) source coordinates in meters

    # Compute basis matrix: same for fixed-/free-orientation
    B = compute_B(src_coords, sigma, threshold_factor)   # (N, N), sigma = 0.001 -- use small kernel width (in meters)
    N = src_coords.shape[0]

    # Handle orientation type
    if n_orient == 1:
        # Fixed-orientation case
        assert L.shape[1] == N, f"L has {L.shape[1]} cols, expected N={N} for fixed-orientation."
        G = L @ B        # (M × N)
        group_size = 1
    elif n_orient == 3:
        # Free-orientation case
        I3 = identity(3, format='csr')
        B_big = kron(I3, B, format='csr')    # (3N, 3N) block-diag(B,B,B)
        G = L @ B_big                        # (M, 3N)
        group_size = 3
    else:
        raise ValueError("n_orient must be 1 (fixed) or 3 (free).")
  
    # Run gamma-MAP on G
    gamma_result = gamma_map(
        L=G,  # Use pseudo-lead field instead of original L
        y=y,
        noise_var=noise_var,
        n_orient=group_size,  # Use appropriate group size: 1 (fixed) or 3 (free)
        max_iter=max_iter,
        tol=tol,
        update_mode=update_mode,
        init_gamma=init_gamma,
        verbose=verbose,
        logger=logger
    )
    c_hat = gamma_result["posterior_mean"]
    active_indices = gamma_result.get("active_indices")
    posterior_cov = gamma_result.get("posterior_cov")
    gamma = gamma_result.get("gamma")
    noise_var_result = gamma_result.get("noise_var", noise_var)

    # Reconstruct sources
    # NOTE: gamma_map already returns FULL c_hat (zeros at inactive indices).
    # NOTE: If n_orient==3, c_hat has shape (N, 3, T); flatten for linear ops.

    if n_orient == 3 and c_hat.ndim == 3:  # Free-orientation
        T = c_hat.shape[-1]
        c_hat_vec = c_hat.reshape(3 * N, T)             # (3N, T)
    else:                                  # Fixed-orientation
        c_hat_vec = c_hat                                # (N, T)

    # posterior mean and covaraince in source space (after B)
    if n_orient == 1:  # fixed-orientation
        x_hat = B @ c_hat_vec                            # (N, T)
        # posterior covariance in source space using ACTIVE block only
        posterior_cov = B[:, active_indices] @ posterior_cov @ B[:, active_indices].T
    else:
        x_hat = B_big @ c_hat_vec                        # (3N, T)
        posterior_cov = B_big[:, active_indices] @ posterior_cov @ B_big[:, active_indices].T
        # optional reshape back to (N, 3, T)
        x_hat = x_hat.reshape(N, 3, -1)

    return {
        "posterior_mean": x_hat,
        "posterior_cov": posterior_cov,
        "noise_var": noise_var_result,
        "active_indices": active_indices,
        "gamma": gamma,
    }


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
    
    Parameters:
    -----------
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
        
    Returns:
    --------
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
    
    Parameters:
    -----------
    G : ndarray, shape (n_chan, n_src * n_orient)
        The original lead-field matrix, after whitening and orientation‐prior scaling.
    n_orient : int
        Number of orientations per source (1 for fixed, 3 for free orientation).
        
    Returns:
    --------
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
    
    Parameters:
    -----------
    other : ndarray, shape (n_chan, n_src * n_orient) or similar
        The matrix to be multiplied with R_sqrt.
    R_sqrt : ndarray
        The square root of the source covariance matrix R. It is either a 1D vector
        (for a diagonal matrix) or a 3D array (for block-diagonal multi-orientation case).
        
    Returns:
    --------
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
    
    Parameters:
    -----------
    sing : array-like, singular values from the SVD.
    n_nzero : int, number of non-zero singular values (typically number of sensors).
    lambda2 : float, regularization parameter.
    
    Returns:
    --------
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
    
    Parameters:
    -----------
    G : ndarray, the lead-field matrix.
    n_orient : int, number of orientations per source.
    loose : float, scaling factor for certain orientations.
    
    Returns:
    --------
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
    
    Parameters:
    -----------
    A : ndarray
        The matrix for which to compute the singular value decomposition.
    full_matrices : bool
        Flag determining if full or reduced SVD is computed.
    
    Returns:
    --------
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
    
    Parameters:
    -----------
    L : ndarray, shape (n_chan, n_src*n_orient)
        The original lead-field matrix.
    lambda2 : float, regularization parameter to stabilize the inversion.
    n_orient : int, the number of orientations per source (1 for fixed orientation, 3 for free orientation).
    whitener : ndarray, the whitening matrix derived from the noise covariance.
    loose : float, parameter for the orientation prior (looseness of the constraints).
    max_iter : int, maximum number of iterations for the iterative fitting procedure.
    
    Returns:
    --------
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
    
    Parameters:
    -----------
    L : ndarray, shape (n_chan, n_src*n_orient)
        The lead-field (forward) matrix mapping sources to sensors.
    y : ndarray, shape (n_chan, n_times) or (n_chan,)
        The sensor data (EEG/MEG recordings) to be inverted.

    n_orient : int
        Number of orientations per source (1 for fixed or 3 for free orientation).
    
    Returns:
    --------
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

        Parameters:
        - solver (callable): The inverse solver function (e.g., gamma_map, eloreta).
        - solver_params (dict, optional): Parameters for the solver function.
        - noise_var (float, optional): Noise variance for the solver.
        - logger (logging.Logger, optional): Logger instance for logging messages.
        - n_orient (int, optional): Number of orientations for the sources.
          Default is 1 (for fixed orientation) or 3 (for free orientation).
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
                    "Updating n_orient from %s to %s based on leadfield shape.",
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

        Parameters:
        - L (np.ndarray): Leadfield matrix of shape (n_sensors, n_sources)
          for fixed orientation or (n_sensors, n_sources, n_orient) for free orientation.
        - y (np.ndarray): Observed EEG/MEG signals of shape (n_sensors, n_times).

        Returns:
        - self: The fitted estimator.
        """
        self.logger.debug("Fitting the solver...")
        self.L_ = self._format_leadfield(L)
        self.y_ = y
        
        return self

    def _get_coef(self, y):
        """
        Internal method to compute the source estimates.

        Parameters:
        - y (np.ndarray): Observed EEG/MEG signals of shape (n_sensors, n_times).

        Returns:
        - x_hat (np.ndarray): Estimated source activity of shape (n_sources, n_times).
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
            # fallback for solvers that do not accept noise_var argument (e.g. joint learning with sflex_gamma_lambda_map())
            return self.solver(**solver_kwargs)
    
    def predict(self, y=None):
        if y is None:
            if not hasattr(self, "y_"):
                raise ValueError("Estimator has not been fitted and no data was provided to predict().")
            y = self.y_
        return self._get_coef(y)

# ==================
# Cross validation
# ==================

def _logdet(A):
    """Compute the logdet of a positive semidefinite matrix."""
    from scipy import linalg

    vals = linalg.eigvalsh(A)
    # avoid negative (numerical errors) or zero (semi-definite matrix) values
    tol = vals.max() * vals.size * np.finfo(np.float64).eps
    vals = np.where(vals > tol, vals, tol)
    return np.sum(np.log(vals))

def logdet_bregman_div_distance_nll(y, Sigma_Y):
    """Compute the log-det Bregman divergence between two matrices."""
    Sigma_Y_inv = np.linalg.inv(Sigma_Y)
    n_features, n_times = y.shape
    Cov_y = y @ y.T / n_times
    log_like = np.mean(np.sum((y.T @ Sigma_Y_inv) * y.T, axis=1))
    log_like -= _logdet(Cov_y @ Sigma_Y_inv)
    out = log_like - n_features
    return out

class SpatialSolver(SourceEstimator):
    """Lightweight sklearn-compatible adaptor around SourceEstimator for CV.

    The constructor must store all input parameters as attributes and must not
    mutate them so that sklearn.clone can work correctly (required by
    GridSearchCV). The class implements a small-fit/predict wrapper around the
    underlying solver so CV can call .fit/.predict as expected.
    """
    def __init__(self, solver, solver_params=None, noise_var=None, n_orient=1,  logger=None):
        super().__init__(solver, solver_params=solver_params, noise_var=noise_var, n_orient=n_orient, logger=logger)

    def fit(self, L, y):
        """Fit by running the underlying solver to produce x_hat and posterior cov.

        Parameters
        ----------
        L : ndarray
            Leadfield matrix (n_sensors, n_sources)
        y : ndarray
            Sensor data (n_sensors, n_times)
        """
        super().fit(L, y)
        self.coef_ = self._get_coef(self.y_)

        return self

    def predict(self, L):
        # Predict sensor data from leadfield L and estimated sources x_hat
        posterior_mean = self.coef_.get("posterior_mean")
        return L @ posterior_mean  # posterior_mean is x_hat

    def score(self, L, y):
        # Simple negative MSE score compatible with sklearn (higher is better)
        y_pred = self.predict(L)
        return -np.mean((y_pred - y) ** 2)

class BaseCVSolver(SourceEstimator):
    def __init__(
        self,
        solver,
        solver_params=None,
        n_orient=1,
        noise_variances=None,
        cv=5,
        n_jobs=1,
        logger=None,
    ):
        super().__init__(
            solver,
            solver_params=solver_params,
            noise_var=None,
            n_orient=n_orient,
            logger=logger,
        )
        self.noise_variances = noise_variances
        self.cv = cv
        self.n_jobs = n_jobs

    def fit(self, L, y):
        return super().fit(L, y)

    def predict(self, y=None):
        if y is None:
            if not hasattr(self, "y_"):
                raise ValueError("Estimator has not been fitted and no data was provided to predict().")
            y = self.y_
        self._get_noise_var(y)
        return self.solver(
            self.L_,
            y,
            noise_var=self.noise_var,
            n_orient=self.n_orient,
            logger=self.logger,
            **(self.solver_params or {}),
        )

class SpatialCVSolver(BaseCVSolver):
    def _get_noise_var(self, y):
        """Sets noise_var attribute with spatial cross-validation."""
        gs = GridSearchCV(
            estimator=SpatialSolver(
                self.solver,
                solver_params=self.solver_params,
                noise_var=self.noise_var,
                n_orient=self.n_orient,
                logger=self.logger,
            ),
            param_grid=dict(noise_var=self.noise_variances),
            scoring="neg_mean_squared_error",
            cv=self.cv,
            n_jobs=self.n_jobs,
            error_score="raise",
        )
        if self.logger is not None:
            self.logger.debug("Running spatial cross-validation...")
        gs.fit(self.L_, y)
        self.grid_search_ = gs
        self.noise_var = gs.best_estimator_.noise_var

class TemporalCVSolver(BaseCVSolver):
    def _get_noise_var(self, y):
        """Sets noise_var attribute with temporal cross-validation."""
        base_solver = SpatialSolver(
            self.solver,
            solver_params=self.solver_params,
            noise_var=self.noise_var,
            n_orient=self.n_orient,
            logger=self.logger,
        )
        
        cv = check_cv(self.cv)
        scores = []
        sensor_eye = np.eye(self.L_.shape[0])
        for noise_var in self.noise_variances:
            noise_cov = noise_var * sensor_eye  # TODO: check if this works for all noise types
            
            solver = clone(base_solver)
            solver.set_params(noise_var=noise_var)
            temporal_cv_scores = []
            for train_idx, test_idx in cv.split(y.T):
                solver.fit(self.L_, y[:, train_idx])
                y_test = y[:, test_idx]

                posterior_mean = solver.coef_.get("posterior_mean")
                X_var = np.mean(posterior_mean ** 2, axis=1)
                Sigma_Y = noise_cov + ((self.L_ * X_var[None, :]) @ self.L_.T)

                temporal_cv_scores.append(
                    logdet_bregman_div_distance_nll(y_test, Sigma_Y)
                )
            scores.append(np.mean(temporal_cv_scores))
        scores = np.asarray(scores)
        # best_idx = int(np.argmax(scores))
        best_idx = int(np.argmin(scores))
        self.noise_var = self.noise_variances[best_idx]


# =================
# Gamma-MAP with Joint Learning
# =================

def _lambda_opt(
    M, G, x_hat_full, posterior_cov_active, active_indices,
    current_lambda, CMinv, update_mode_noise
):
    """
    Update noise variance lambda (NORMALIZED scale) using EM or Convex Bounding.

    Parameters (all NORMALIZED scale)
    Note: Ka = len(active_indices)
    --------------------------------
    M : ndarray, shape (M,T)
        Normalized sensor data.
    G : ndarray, shape (M,Ka)
        Active leadfield (columns for active sources).
    x_hat_full : ndarray, shape (N,T)
        Full posterior mean in coefficient space (zeros for inactive).
    posterior_cov_active : ndarray, shape (Ka,Ka)
        Posterior covariance on active set.
    active_indices : ndarray, shape (Ka,)
        Active indices into full source set.
    current_lambda : ndarray, shape (M,)
        Current lambda (normalized).
    CMinv : ndarray, shape (M,M)
        Inverse sensor covariance.
    update_mode_noise : int
        1 = EM update, 2 = Convex Bounding update rule.

    Returns
    -------
    lambda_new : ndarray, shape (M,)
        Updated lambda (normalized).
    """
    n_sensors = M.shape[0]
    n_sources = x_hat_full.shape[0]
    n_active = len(active_indices)
    lambda_new = np.zeros(n_sensors)

    # Full forward operator (with zeros at inactive columns)
    G_full = np.zeros((n_sensors, n_sources))

    # Ensure G has exactly n_active columns
    if G.shape[1] != n_active:
        if G.shape[1] > n_active:
            G = G[:, :n_active]
        elif G.shape[1] < n_active:
            G_padded = np.zeros((n_sensors, n_active))
            G_padded[:, :G.shape[1]] = G
            G = G_padded

    G_full[:, active_indices] = G

    # Full posterior covariance (zeros for inactive)
    posterior_cov_full = np.zeros((n_sources, n_sources))
    idx_active = np.ix_(active_indices, active_indices)
    posterior_cov_full[idx_active] = posterior_cov_active

    if update_mode_noise == 1:
        for m in range(n_sensors):
            residual = M[m] - np.dot(G_full[m], x_hat_full)
            residual_term = np.mean(residual ** 2)
            cov_term = np.dot(G_full[m], np.dot(posterior_cov_full, G_full[m]))
            lambda_new[m] = residual_term + cov_term

    elif update_mode_noise == 2:
        for m in range(n_sensors):
            residual = M[m] - np.dot(G_full[m], x_hat_full)
            numerator = np.mean(residual ** 2)
            denominator = CMinv[m, m]
            if denominator > 1e-16:
                lambda_new[m] = np.sqrt(numerator / denominator)
            else:
                lambda_new[m] = current_lambda[m]
    else:
        raise ValueError("Noise update mode must be 1 (EM) or 2 (Convex Bounding)")

    lambda_new = np.maximum(lambda_new, 1e-16)
    return lambda_new

def _gamma_map_opt_with_lambda(
    M,
    G,
    sigma_squared,           # ORIGINAL scale fallback (only used if init_lambda is None AND learn_lambda=False)
    maxit=1000,
    tol=1e-6,
    update_mode=2,
    group_size=1,
    init_gamma=None,
    init_lambda=None,        # ORIGINAL scale init (or None)
    learn_lambda=True,
    update_mode_noise=2,
    track_history=True,
    verbose=False,
):
    """
    Core Gamma-MAP optimizer with optional alternating lambda updates.

    Parameters
    ----------
    M : ndarray, shape (M,T)
        Sensor data (original scale).
    G : ndarray, shape (M,N)
        Leadfield (original scale).
    sigma_squared : float or ndarray, shape (M,)
        Used ONLY when learn_lambda=False and init_lambda is None (fixed-lambda fallback).
    maxit : int
        Maximum iterations.
    tol : float
        Tolerance on gamma relative change (stopping criterion).
    update_mode : {1,2,3}
        Gamma update rule variant.
    group_size : int
        Grouping size (1 fixed ori, 3 free ori grouping).
    init_gamma : None or ndarray, shape (N,)
        If None -> ones(N).
    init_lambda : None or scalar or ndarray, shape (M,)
        Original-scale initialization.
        If None:
          - learn_lambda=True  -> ones(M)
          - learn_lambda=False -> derived from sigma_squared/noise_var
    learn_lambda : bool
        True: alternate gamma/lambda updates. False: keep lambda fixed.
    update_mode_noise : {1,2}
        Lambda update rule variant.
    track_history : bool
        If True, returns n_active_hist and err_hist.
    verbose : bool
        Print progress every ~50 iters.

    Returns
    -------
    x_active : ndarray, shape (Ka,T)
        Posterior mean for active sources (ORIGINAL scale).
    active_indices : ndarray, shape (Ka,)
        Active indices.
    posterior_cov_active : ndarray, shape (Ka,Ka)
        Posterior covariance for active set (ORIGINAL scale).
    gammas_full : ndarray, shape (N,)
        Full gamma vector (inactive entries are 0).
    lambda_final : ndarray, shape (M,)
        Full lambda vector (ORIGINAL scale).
    hist : dict
        If track_history:
          - n_active_hist : list[int]
          - err_hist      : list[float]
        else empty dict.
    """
    G = np.asarray(G, dtype=float).copy()
    M = np.asarray(M, dtype=float).copy()
    if M.ndim == 1:
        M = M[:, None]

    n_sources = G.shape[1]
    n_sensors, n_times = M.shape
    eps = np.finfo(float).eps

    # -------------------------
    # Normalization
    # -------------------------
    M_normalize_constant = np.linalg.norm(M @ M.T, ord="fro")
    M /= (np.sqrt(M_normalize_constant) + eps)

    G_normalize_constant = np.linalg.norm(G, ord=np.inf)
    G /= (G_normalize_constant + eps)

    if n_sources % group_size != 0:
        raise ValueError("Number of sources has to be evenly dividable by group_size")

    # -------------------------
    # Init gamma: None -> ones
    # -------------------------
    
    if init_gamma is None:
        gammas_full_old = np.ones(n_sources, dtype=np.float64)
    elif isinstance(init_gamma, (float, np.float64, int, np.int64)):
        gammas_full_old = np.full((n_sources,), init_gamma, dtype=np.float64)
    else:
        gammas_full_old = np.asarray(init_gamma, dtype=np.float64).copy()
        if gammas_full_old.size != n_sources:
            raise ValueError("init_gamma must have length equal to number of sources (G.shape[1]).")

    # -------------------------
    # Init lambda (requested rule + baseline fallback)
    # -------------------------
    if init_lambda is None:
        if not learn_lambda:
            # fixed-lambda baseline uses sigma_squared/noise_var as fixed lambda
            if np.isscalar(sigma_squared):
                lambda_orig = np.full(n_sensors, float(sigma_squared), dtype=np.float64)
            else:
                lambda_orig = np.asarray(sigma_squared, dtype=np.float64).copy()
        else:
            # adaptive mode default init is ones(M)
            lambda_orig = np.ones(n_sensors, dtype=np.float64)
    else:
        if np.isscalar(init_lambda):
            lambda_orig = np.full(n_sensors, float(init_lambda), dtype=np.float64)
        else:
            lambda_orig = np.asarray(init_lambda, dtype=np.float64).copy()

    if lambda_orig.size != n_sensors:
        raise ValueError("init_lambda (or sigma_squared vector) must have length n_sensors (G.shape[0]).")

    # ORIGINAL -> NORMALIZED
    current_lambda = lambda_orig / (M_normalize_constant + eps)

    # -------------------------
    # History (NO mean tracking)
    # -------------------------
    hist = {}
    if track_history:
        hist["n_active_hist"] = []
        hist["err_hist"] = []

    denom_fun = np.sqrt if update_mode == 2 else (lambda x: x)

    active_indices = np.arange(n_sources)
    gammas_active_new = None
    posterior_cov_active = None
    CMinv = None
    A = None
    G_CMinvG = None

    for itno in range(int(maxit)):
        gammas_active = gammas_full_old[active_indices]
        gammas_active[np.isnan(gammas_active)] = 0.0

        # prune
        keep = np.abs(gammas_active) > eps
        active_indices = active_indices[keep]
        gammas_active = gammas_active[keep]

        if active_indices.size == 0:
            break

        G_active = G[:, active_indices]

        # CM = G diag(gamma) G^T + diag(lambda)
        CM = (G_active * gammas_active[None, :]) @ G_active.T
        np.fill_diagonal(CM, CM.diagonal() + current_lambda)

        # invert
        try:
            U, S, _ = linalg.svd(CM, full_matrices=False)
            CMinv = (U / (S[None, :] + eps)) @ U.T
        except linalg.LinAlgError:
            CMinv = linalg.pinv(CM)

        CMinvG = CMinv @ G_active
        A = CMinvG.T @ M  # (Ka,T)

        # gamma update
        if update_mode == 1:
            numer = gammas_active ** 2 * np.mean((A * A.conj()).real, axis=1)
            denom = gammas_active * np.sum(G_active * CMinvG, axis=0)
        elif update_mode == 2:
            numer = gammas_active * np.sqrt(np.mean((A * A.conj()).real, axis=1))
            denom = np.sum(G_active * CMinvG, axis=0)
        elif update_mode == 3:
            denom = None
            numer = gammas_active ** 2 * np.mean((A * A.conj()).real, axis=1) + gammas_active * (
                1 - gammas_active * np.sum(G_active * CMinvG, axis=0)
            )
        else:
            raise ValueError("Invalid value for update_mode")

        if group_size == 1:
            if denom is None:
                gammas_active_new = numer
            else:
                gammas_active_new = numer / np.maximum(denom_fun(denom), np.finfo(float).eps)
        else:
            numer_comb = np.sum(numer.reshape(-1, group_size), axis=1)
            if denom is None:
                gammas_comb = numer_comb
            else:
                denom_comb = np.sum(denom.reshape(-1, group_size), axis=1)
                gammas_comb = numer_comb / denom_fun(denom_comb)
            gammas_active_new = np.repeat(gammas_comb / group_size, group_size)

        gammas_active_new = np.maximum(gammas_active_new, 0.0)

        # posterior cov (normalized)
        G_CMinvG = G_active.T @ CMinvG
        posterior_cov_active = (
            np.diag(gammas_active_new)
            - gammas_active_new[:, None] * G_CMinvG * gammas_active_new
        )

        # x_hat_full (normalized) for lambda update
        x_hat_full = np.zeros((n_sources, n_times))
        x_hat_full[active_indices] = gammas_active_new[:, None] * A

        # lambda update (only if learn_lambda=True)
        if learn_lambda:
            current_lambda = _lambda_opt(
                M=M,
                G=G_active,
                x_hat_full=x_hat_full,
                posterior_cov_active=posterior_cov_active,
                active_indices=active_indices,
                current_lambda=current_lambda,
                CMinv=CMinv,
                update_mode_noise=update_mode_noise,
            )

        # full gamma
        gammas_full = np.zeros(n_sources, dtype=np.float64)
        gammas_full[active_indices] = gammas_active_new

        # convergence (gamma only)
        err = np.sum(np.abs(gammas_full - gammas_full_old)) / (np.sum(np.abs(gammas_full_old)) + eps)
        gammas_full_old = gammas_full

        if track_history:
            hist["n_active_hist"].append(int(active_indices.size))
            hist["err_hist"].append(float(err))

        if verbose and (itno % 50 == 0 or err < tol):
            print(
                f"iter={itno:5d} | n_active={active_indices.size:4d} | err={err:.3e} "
                f"| mean(lambda)={np.mean(current_lambda)*M_normalize_constant:.3e}"
            )

        if err < tol or active_indices.size == 0:
            break

    # all pruned
    if gammas_active_new is None or active_indices.size == 0:
        x_active = np.zeros((0, n_times))
        posterior_cov_out = np.zeros((0, 0))
        gammas_full = np.zeros(n_sources, dtype=np.float64)
        lambda_final = current_lambda * M_normalize_constant
        return x_active, active_indices, posterior_cov_out, gammas_full, lambda_final, hist

    # undo normalization (ORIGINAL scale)
    n_const = np.sqrt(M_normalize_constant) / (G_normalize_constant + eps)
    x_active = n_const * gammas_active_new[:, None] * A

    posterior_cov_out = (
        np.diag(gammas_active_new)
        - gammas_active_new[:, None] * G_CMinvG * gammas_active_new
    )
    posterior_cov_out = (n_const ** 2) * posterior_cov_out

    lambda_final = current_lambda * M_normalize_constant

    return x_active, active_indices, posterior_cov_out, gammas_full, lambda_final, hist

def gamma_lambda_map(
    L,
    y,
    noise_var,
    init_gamma=None,
    init_lambda=None,
    n_orient=1,
    max_iter=1000,
    tol=1e-15,
    update_mode=2,
    learn_lambda=True,
    update_mode_noise=2,
    track_history=True,
    verbose=False,
):
    """
    Non-sFLEX gamma–lambda estimator.

    Parameters
    ----------
    L : ndarray, shape (M,N)
        Leadfield.
    y : ndarray, shape (M,T)
        Sensor data.
    noise_var : float or ndarray, shape (M,)
        Used ONLY if learn_lambda=False and init_lambda is None (fixed-lambda fallback).
    init_gamma : None or ndarray, shape (N,)
        None -> ones(N).
    init_lambda : None or scalar or ndarray, shape (M,)
        Original-scale lambda init.
        None -> ones(M) if learn_lambda=True
        None -> derived from noise_var if learn_lambda=False
    learn_lambda : bool
        Joint learning (True) or fixed lambda (False).
    track_history : bool
        If True, returns n_active_hist and err_hist.

    Returns
    -------
    out : dict with keys
      - x_active : (Ka,T)
      - posterior_cov_active : (Ka,Ka)
      - active_indices : (Ka,)
      - gammas_full : (N,)
      - lambda_final : (M,)
      - lambda_mean : float  (homoscedastic estimate = mean(lambda_final))
      - (optional) n_active_hist, err_hist
    """
    y = np.asarray(y, dtype=float)
    if y.ndim == 1:
        y = y[:, None]
    L = np.asarray(L, dtype=float)

    x_active, active_idx, cov_active, gammas_full, lambda_final, hist = _gamma_map_opt_with_lambda(
        M=y,
        G=L,
        sigma_squared=noise_var,
        maxit=max_iter,
        tol=tol,
        update_mode=update_mode,
        group_size=n_orient,
        init_gamma=init_gamma,
        init_lambda=init_lambda,
        learn_lambda=learn_lambda,
        update_mode_noise=update_mode_noise,
        track_history=track_history,
        verbose=verbose,
    )

    out = {
        "x_active": x_active,
        "active_indices": active_idx,
        "posterior_cov_active": cov_active,
        "gammas_full": gammas_full,
        "lambda_final": lambda_final,
        "lambda_mean": float(np.mean(lambda_final)),
    }
    if track_history:
        out.update(hist)
    return out

def sflex_gamma_lambda_map(
    L,
    y,
    fwd_path,
    noise_var,
    sigma=0.001,
    init_gamma=None,
    init_lambda=None,
    update_mode_noise=2,
    n_orient=1,
    max_iter=10000,
    tol=1e-15,
    update_mode=2,
    learn_lambda=True,
    threshold_factor=3.0,
    track_history=True,
    logger=None,
    verbose=False,
):
    """
    sFLEX + gamma_lambda_map.

    Notes
    -----
    - Builds pseudo leadfield: G = L @ B (fixed ori) or G = L @ (I3 ⊗ B) (free ori)
    - Runs gamma–lambda in coefficient space.
    - Maps posterior mean/cov back to source space.

    Parameters
    ----------
    L : ndarray, shape (M,N_sub)
        Leadfield for subset of sources.
    y : ndarray, shape (M,T)
        Sensor data.
    fwd_path : str
        MNE forward solution path (for source coordinates).
    sigma, threshold_factor : float
        sFLEX basis params.
    noise_var : float or ndarray, shape (M,)
        Used ONLY if learn_lambda=False and init_lambda is None.
    init_gamma, init_lambda, learn_lambda : see gamma_lambda_map.

    Returns
    -------
    out : dict with keys
      - posterior_mean_full : (N_sub,T) or (N_sub,3,T)
      - posterior_cov_full  : (N_sub,N_sub) or (3N_sub,3N_sub)
      - active_indices      : active indices in coefficient space
      - gammas_full         : full gamma vector in coefficient space
      - lambda_final        : (M,) sensor-space lambda (original scale)
      - lambda_mean         : float
      - (optional) n_active_hist, err_hist
    """
    fwd_path = f"{fwd_path}-fwd.fif"
    fwd = mne.read_forward_solution(fwd_path, verbose="error")

    if n_orient == 2 and fwd['source_ori'] == FIFF.FIFFV_MNE_FREE_ORI:
        fwd = mne.convert_forward_solution(fwd, force_fixed=True)
        
    # src_coords = fwd['src'][0]['rr']  # (N, 3) source locations in meters
    src_coords = fwd['source_rr']  # (N, 3) source coordinates in meters   

    B = compute_B(src_coords, sigma, threshold_factor)
    N = src_coords.shape[0]

    if n_orient == 1:
        G = L @ B
        group_size = 1
        B_big = None
    elif n_orient == 3:
        I3 = identity(3, format='csr')
        B_big = kron(I3, B, format='csr')
        G = L @ B_big
        group_size = 3
    else:
        raise ValueError("n_orient must be 1 or 3.")

    res = gamma_lambda_map(
        L=G,
        y=y,
        noise_var=noise_var,
        init_gamma=init_gamma,
        init_lambda=init_lambda,
        n_orient=group_size,
        max_iter=max_iter,
        tol=tol,
        update_mode=update_mode,
        learn_lambda=learn_lambda,
        update_mode_noise=update_mode_noise,
        track_history=track_history,
        verbose=verbose,
    )

    active_idx = res["active_indices"]
    c_active = res["x_active"]  # (Ka,T)

    # FULL coefficient mean
    c_full = np.zeros((G.shape[1], y.shape[1]))
    c_full[active_idx] = c_active

    # Map back to source space
    if n_orient == 1:
        posterior_mean_full = (B @ c_full)  # (N,T)
        posterior_cov_full = B[:, active_idx] @ res["posterior_cov_active"] @ B[:, active_idx].T  # (N,N)
    else:
        posterior_mean_full = (B_big @ c_full)  # (3N,T)
        posterior_cov_full = B_big[:, active_idx] @ res["posterior_cov_active"] @ B_big[:, active_idx].T  # (3N,3N)
        posterior_mean_full = posterior_mean_full.reshape(N, 3, -1)

    # Final gamma vector (normalized parameterization) and its norm
    gamma_norm = np.linalg.norm(res["gammas_full"])
    
    out = {
        "posterior_mean": posterior_mean_full,
        "posterior_cov": posterior_cov_full,
        "active_indices": active_idx,
        "gamma": gamma_norm,
        "gammas_full": res["gammas_full"],
        "lambdas": res["lambda_final"],
        "lambda_mean": res["lambda_mean"],
    }
    if track_history:
        out["n_active_hist"] = res.get("n_active_hist", [])
        out["err_gamma_hist"] = res.get("err_hist", [])
    return out
