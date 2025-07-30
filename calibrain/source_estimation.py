import numpy as np
import pandas as pd
from scipy import linalg
from warnings import warn
from sklearn.base import BaseEstimator, RegressorMixin
from scipy import linalg


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
        warn("\nConvergence NOT reached !\n")

    # undo normalization and compute final posterior mean
    n_const = np.sqrt(M_normalize_constant) / G_normalize_constant
    x_active = n_const * init_gamma[:, None] * A

    # Compute the posterior convariance matrix as in eq. (2.10) in Hashemi, Ali. "Advances in hierarchical Bayesian learning with applications to neuroimaging." (2023).
    # pos_cov =  np.diag(init_gamma) - init_gamma[:, np.newaxis] * G_CMinvG * init_gamma
    posterior_cov = np.diag(init_gamma) - init_gamma[:, np.newaxis] * G.T @ CMinv @ G * init_gamma 
    # A similar approach can be implmented (as Large_gamma is interpreted as adiagonal matrix with small_gammas:
    # posterior_cov = np.diag(init_gamma) - np.diag(init_gamma) @ G.T @ CMinv @ G @ np.diag(init_gamma)
    
    return x_active, active_indices, posterior_cov

def eloreta(L, y, **kwargs):
    raise NotImplementedError("The eloreta solver is not yet implemented.")

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
        x_hat, active_indices, posterior_cov = self.solver(self.L_, y, noise_var, logger=self.logger, **self.solver_params, n_orient=self.n_orient)
        
        return x_hat, active_indices, posterior_cov