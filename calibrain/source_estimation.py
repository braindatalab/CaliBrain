import numpy as np
import pandas as pd
from scipy import linalg
import warnings
from sklearn.base import BaseEstimator, RegressorMixin
from scipy import linalg

from functools import partial
from mne.utils import sqrtm_sym, eigh
from scipy.stats import chi2, norm
from matplotlib import cm

# ===================
# GAMMA-MAP Functions
# ===================

def gamma_map(
    L,
    y,
    noise_var=None,
    noise_type="oracle",
    n_orient=1,
    max_iter=1000,
    tol=1e-15,
    update_mode=2,
    # threshold=1e-5,
    init_gamma=None,
    verbose=True,
    logger=None,
):
    # sigma_squared: noise variance = diagonal of the covariance matrix, where all diagonal elements are equal.
    if noise_type == "oracle":
        noise_cov = noise_var * np.eye(L.shape[0])
    
    # TODO: check whether we still need this
    if init_gamma is None:
        init_gamma = np.ones(L.shape[1], dtype=np.float64)
    elif isinstance(init_gamma, (float, np.float64, int, np.int64)):
        init_gamma = np.full((L.shape[1],), init_gamma, dtype=np.float64)
    elif len(init_gamma) == 2 and isinstance(init_gamma, tuple):
        init_gamma = np.linspace(init_gamma[0], init_gamma[1], num=L.shape[1])
    else:
        raise ValueError("init_gamma should be a float, a tuple of two floats, or a list of floats.")

    x_hat_, active_indices, posterior_cov = _gamma_map_opt(
        y,
        L,
        sigma_squared=noise_var,
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

    return x_hat, active_indices, posterior_cov

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
            # logger.info(
            #     "Iteration: %d\t active set size: %d\t convergence: "
            #     "%0.3e" % (itno, len(init_gamma), err)
            # )
            last_size = len(init_gamma)

        if breaking:
            break

    if itno < maxit - 1:
        logger.info(
            "Iteration: %d\t active set size: %d\t convergence: "
            "%0.3e" % (itno, len(init_gamma), err)
        )
        logger.info("\nConvergence reached !\n")
    else:
        logger.info(
            "Iteration: %d\t active set size: %d\t convergence: "
            "%0.3e" % (itno, len(init_gamma), err)
        )
        warnings.warn("\nConvergence NOT reached !\n")

    # undo normalization and compute final posterior mean
    n_const = np.sqrt(M_normalize_constant) / G_normalize_constant
    x_active = n_const * init_gamma[:, None] * A

    # Compute the posterior convariance matrix as in eq. (2.10) in Hashemi, Ali. "Advances in hierarchical Bayesian learning with applications to neuroimaging." (2023).
    # pos_cov =  np.diag(init_gamma) - init_gamma[:, np.newaxis] * G_CMinvG * init_gamma
    posterior_cov = np.diag(init_gamma) - init_gamma[:, np.newaxis] * G.T @ CMinv @ G * init_gamma 
    # A similar approach can be implmented (as Large_gamma is interpreted as adiagonal matrix with small_gammas:
    # posterior_cov = np.diag(init_gamma) - np.diag(init_gamma) @ G.T @ CMinv @ G @ np.diag(init_gamma)
    
    return x_active, active_indices, posterior_cov

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

def compute_eloreta_kernel(L, *, lambda2, n_orient, whitener, loose=1.0, max_iter=20):
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
        warnings.warn("eLORETA weight fitting did not converge (>= %s)" % eps)
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

def eloreta(L, y, noise_var=None, noise_type="oracle",  n_orient=1, verbose=True, logger=None):
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
    if noise_type == "oracle":
        noise_cov = noise_var * np.eye(L.shape[0])
    
    # Create the whitening matrix from the noise covariance:
    # Typically computed as the inverse of the square root of the covariance.
    whitener = linalg.inv(linalg.sqrtm(noise_cov))
    
    # Whiten both the sensor data and the lead-field matrix.
    y = whitener @ y
    L = whitener @ L

    # Compute the eLORETA kernel and the posterior source covariance using the helper.
    # alpha is lambda2 = noise_var
    K, Sigma = compute_eloreta_kernel(L, lambda2= noise_var, n_orient=n_orient, whitener=whitener)
    
    # Compute the mean source estimates.
    x = K @ y # get the source time courses with simple dot product

    # If using free orientation sources (n_orient > 1), reshape the output.
    if n_orient > 1:
        x = x.reshape((-1, n_orient, x.shape[1]))

    active_indices = np.arange(Sigma.shape[0])  # All sources are active in eLORETA
    return x, active_indices, Sigma  # Return source estimates, all active indices, and posterior covariance


class SourceEstimator(BaseEstimator, RegressorMixin):
    def __init__(self, solver, solver_params=None, n_orient=1, logger=None):
        """
        Initialize the SourceEstimator class.

        Parameters:
        - solver (callable): The inverse solver function (e.g., gamma_map, eloreta).
        - solver_params (dict, optional): Parameters for the solver function.
        - logger (logging.Logger, optional): Logger instance for logging messages.
        - n_orient (int, optional): Number of orientations for the sources.
          Default is 1 (for fixed orientation) or 3 (for free orientation).
        """
        self.solver = solver
        self.solver_params = solver_params if solver_params else {}
        self.logger = logger
        # self.cov = cov
        self.n_orient = n_orient

    def fit(self, L, y):
        """
        Fit the inverse solver to the data.

        Parameters:
        - L (np.ndarray): Leadfield matrix of shape (n_sensors, n_sources).
        - y (np.ndarray): Observed EEG/MEG signals of shape (n_sensors, n_times).

        Returns:
        - self: The fitted estimator.
        """
        self.logger.info("Fitting the solver...")
        self.L_ = L
        self.y_ = y
        return self

    def predict(self, y=None, noise_var=None):
        """
        Predict the source activity given the observed signals.

        Parameters:
        - y (np.ndarray, optional): Observed EEG/MEG signals of shape (n_sensors, n_times). If None, uses the signals provided during `fit`.
        - noise_var (float): Noise variance.

        Returns:
        - x_hat (np.ndarray): Estimated source activity of shape (n_sources, n_times).
        - active_indices (np.ndarray): Indices of active sources.
        - posterior_cov (np.ndarray): Posterior covariance matrix of estimated sources.
        """
        if not hasattr(self, "L_") or not hasattr(self, "y_"):
            raise ValueError("The estimator must be fitted with `fit(L, y)` before calling `predict()`.")
        
        # enable the use to pass y for inference
        if y is None: 
            y = self.y_

        # Apply the solver
        self.logger.info("Estimating sources...")
        x_hat, active_indices, posterior_cov = self.solver(self.L_, y, noise_var, logger=self.logger, n_orient=self.n_orient, **self.solver_params)
        
        return x_hat, active_indices, posterior_cov