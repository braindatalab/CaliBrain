# -*- coding: utf-8 -*-
"""
Created on Tue Apr 15 13:50:51 2025

@author: Ismail Huseynov
"""


# =============================================================================
# IMPORTS
# =============================================================================
# Import essential libraries

import numpy as np
import matplotlib.pyplot as plt
from scipy import linalg
from functools import partial
import warnings
from mne.utils import sqrtm_sym, eigh
from scipy.stats import chi2, norm
from matplotlib import cm

# =============================================================================
# LEAD FIELD MATRIX LOADING & DATA SIMULATION
# =============================================================================
def add_noise(cov_type, y, alpha, rng, n_sensors, n_times):
    """
    Add noise to the observation data.
    """
    # Use the diagonal case: a scaled identity matrix.
    cov = 1e-2 * np.diag(np.ones(n_sensors))
    signal_norm = np.linalg.norm(y, "fro")
    noise = rng.multivariate_normal(np.zeros(n_sensors), cov, size=n_times).T
    noise_norm = np.linalg.norm(noise, "fro")
    scale_factor = ((1 - alpha) / alpha) * (signal_norm / noise_norm)
    noise_scaled = scale_factor * noise
    noise_cov_scaled = cov * (scale_factor ** 2)
    noise_variance_scaled = 1e-2 * scale_factor
    y += noise_scaled
    return y, noise_cov_scaled, noise_variance_scaled, noise_scaled

# =============================================================================
# LEAD FIELD MATRIX LOADING & DATA SIMULATION (FIXED)
# =============================================================================
def get_data(cov_type, path_to_leadfield, n_sensors=50, n_times=10, n_sources=200,
             nnz=3, orientation_type="fixed", alpha=0.01, seed=None):
    """
    Simulate data and load the lead field matrix with proper shape handling
    """
    n_orient = 3 if orientation_type == "free" else 1
    rng = np.random.RandomState(seed)
    lead_field = np.load(path_to_leadfield, allow_pickle=True)
    L = lead_field["lead_field"]
    
    # Handle fixed orientation case
    if orientation_type == "fixed":
        # Convert 3D lead field to 2D by selecting first orientation
        if L.ndim == 3:
            L = L[:, :, 0]  # Shape becomes (n_sensors, n_sources)
        n_sensors, n_sources = L.shape
        
        # Simulate sources with correct dimensions
        x = np.zeros((n_sources, n_times))
        idx = rng.choice(n_sources, size=nnz, replace=False)
        x[idx] = rng.randn(nnz, n_times)
        y = L @ x

    # Rest of function remains the same
    y, noise_cov_scaled, noise_variance_scaled, noise_scaled = add_noise(
        cov_type=cov_type, y=y, alpha=alpha, rng=rng, 
        n_sensors=n_sensors, n_times=n_times)
    
    return y, L, x, noise_cov_scaled, noise_variance_scaled, noise_scaled



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




def eloreta(L, y, cov=1, alpha = 1/9, n_orient=1):
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
    cov : float or ndarray
        Noise covariance matrix; if a scalar is provided, it is converted to
        a diagonal matrix scaled by alpha.
    alpha : float
        Scaling factor for the noise covariance (if cov is provided as a scalar).
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
    # If noise covariance is provided as a scalar, convert to full covariance.
    if isinstance(cov, (float, int)):
        cov = alpha * np.eye(L.shape[0])
    # Create the whitening matrix from the noise covariance:
    # Typically computed as the inverse of the square root of the covariance.
    whitener = linalg.inv(linalg.sqrtm(cov))
    # Whiten both the sensor data and the lead-field matrix.
    y = whitener @ y
    L = whitener @ L

    # Compute the eLORETA kernel and the posterior source covariance using the helper.
    # alpha is lambda2
    K, Sigma = compute_eloreta_kernel(L, lambda2= alpha, n_orient=n_orient, whitener=whitener)
    # Compute the mean source estimates.
    x = K @ y # get the source time courses with simple dot product

    # If using free orientation sources (n_orient > 1), reshape the output.
    if n_orient > 1:
        x = x.reshape((-1, n_orient, x.shape[1]))

    return x, Sigma # Return source estimates and posterior covariance


# =============================================================================
# RUN eLORETA (Fixed Orientation)
# =============================================================================
cov_type = 'diag' # we use scaled identity matrix
leadfield_path = r"your_leadfield_path.npz" # Keep the consistency between the sizes!
n_sensors = 64
n_times = 1000
n_sources = 1284
nnz = 5  # Number of active sources
alpha = 0.01 # SNR = -40 dB 
orientation_type = 'fixed'  # Changed to fixed orientation
seed = 42 # stochastic parameter

y, L, x, noise_cov_scaled, noise_variance_scaled, noise_scaled = get_data(
    cov_type, leadfield_path, n_sensors, n_times, n_sources, nnz,
    orientation_type, alpha, seed)
print("Sensor Data (y) Shape:", y.shape)  # Should be (64, 1000)
print("Lead Field (L) Shape:", L.shape)  # Should be (64, 1284)
print("Ground Truth Sources (x) Shape:", x.shape)  # Should be (1284, 1000)



# Run eLORETA with fixed orientation: n_orient=1
x_hat, posterior_cov = eloreta(L, y, cov=noise_cov_scaled, 
                              alpha=noise_variance_scaled, n_orient=1)  

print("Source Estimates (shape):", x_hat.shape)  # (1284, 1000)
print("Posterior Covariance (shape):", posterior_cov.shape)  # (1284, 1284)

