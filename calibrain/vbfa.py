import numpy as np
from numpy.linalg import svd, inv
import matplotlib.pyplot as plt

def _svd_sqrt(mat: np.ndarray, nl: int) -> np.ndarray:
    """
    Compute the initial mixing matrix b using the SVD of the data covariance matrix.
    """
    p, d, _ = svd(mat)
    return p[:, :nl] @ np.diag(np.sqrt(d[:nl]))

class VBFA:
    """
    Variational Bayes Factor Analysis with regularized covariance estimation.
    Reference: H. T. Attias, "Golden Metallic" (2005).
    """
    def __init__(self, nl: int, nem: int = 50, plot: bool = False):
        """
        Parameters:
        nl  : int   - Number of latent factors
        nem : int   - Number of EM iterations
        plot: bool  - Flag to enable plotting during optimization
        """
        self.nl = nl
        self.nem = nem
        self.plot = plot
        # Model parameters
        self.b = None
        self.lam = None
        self.bet = None
        # Sufficient statistics
        self.likelihood = None

    def fit(self, y: np.ndarray) -> None:
        """
        Fit the VBFA model to the data.

        Parameters:
        y : np.ndarray, shape (nk, nt)
            Data matrix (observations x time points).
        """
        nk, nt = y.shape
        # Precompute data covariance
        ryy = y @ y.T
        # Initialize parameters
        self.b = _svd_sqrt(ryy / nt, self.nl)
        self.lam = 1.0 / np.diag(ryy / nt)
        self.bet = np.ones(self.nl)
        # Tracking
        self.likelihood = np.zeros(self.nem)
        rbb = np.eye(self.nl) / nt

        for i in range(self.nem):
            # E-step
            gam = self._compute_gamma(rbb)
            igam = inv(gam)
            ubar = igam @ self.b.T @ np.diag(self.lam) @ y
            ryu = y @ ubar.T
            ruu = ubar @ ubar.T + nt * igam
            # Compute likelihood
            self.likelihood[i] = self._compute_likelihood(y, ryu, ruu, gam)
            # M-step updates
            self.bet = self._update_bet(ruu)
            self.b = self._update_b(ryu, ruu)
            self.lam = self._update_lam(ryy, ryu)
            # Update regularization term
            rbb = inv(ruu + np.diag(self.bet))
            # Optional plotting
            if self.plot:
                self._plot(i)

        # Final outputs
        self.weight = (self.b @ igam @ self.b.T @ 
                      np.diag(self.lam))
        self.sig = self.b @ self.b.T + np.diag(1.0 / self.lam)
        self.yc = self.b @ ubar
        self.cy = (self.b @ ruu @ self.b.T + 
                  np.diag(np.diag(self.sig) * 
                         np.trace(ruu @ inv(ruu + np.diag(self.bet)))))
        self.mlike = self.likelihood[-1]

    def _compute_gamma(self, rbb: np.ndarray) -> np.ndarray:
        dlam = np.diag(self.lam)
        return (self.b.T @ dlam @ self.b + 
                np.eye(self.nl) + 
                self.b.shape[0] * rbb)

    def _compute_likelihood(self, y: np.ndarray, ryu: np.ndarray, 
                          ruu: np.ndarray, gam: np.ndarray) -> float:
        nk, nt = y.shape
        _, d, _ = svd(gam)
        ldgam = np.sum(np.log(d))
        temp1 = (-0.5 * ldgam + 
                0.5 * np.sum(np.sum(ubar * (gam @ ubar), axis=0)) / nt)
        temp2 = (0.5 * np.sum(np.log(self.lam)) - 
                0.5 * (self.lam @ np.mean(y**2, axis=1)))
        f3 = (0.5 * self.nl * np.sum(np.log(self.lam)) + 
              0.5 * nk * np.sum(np.log(self.bet)) - 
              0.5 * np.trace(self.b.T @ np.diag(self.lam) @ 
                            self.b @ np.diag(self.bet)))
        return temp1 + temp2 + f3 / nt

    def _update_bet(self, ruu: np.ndarray) -> np.ndarray:
        return 1.0 / (np.diag(self.b.T @ np.diag(self.lam) @ self.b) / 
                     self.b.shape[0] + np.diag(inv(ruu)))

    def _update_b(self, ryu: np.ndarray, ruu: np.ndarray) -> np.ndarray:
        return ryu @ inv(ruu + np.diag(self.bet))

    def _update_lam(self, ryy: np.ndarray, ryu: np.ndarray) -> np.ndarray:
        nk = ryy.shape[0]
        ilam = np.diag(ryy - self.b @ ryu.T) / (yk.shape[1] + self.nl)
        return 1.0 / ilam

    def _plot(self, i: int) -> None:
        fig, axes = plt.subplots(3, 3, figsize=(10, 8))
        axes[0, 0].plot(self.likelihood[: i+1])
        axes[0, 0].set_title('Likelihood')
        axes[1, 0].plot(np.sqrt(np.vstack([
            np.mean(self.b**2, axis=0), 
            1.0 / self.bet
        ]).T))
        axes[1, 0].set_title('1/bet')
        axes[2, 0].plot(1.0 / self.lam)
        axes[2, 0].set_title('1/lam')
        # Additional plots can be added similarly
        plt.tight_layout()
        plt.draw()
        plt.pause(0.01)

if __name__ == '__main__':
    # Example usage:
    # y = np.random.randn(64, 1000)
    # model = VBFA(nl=5, nem=50, plot=True)
    # model.fit(y)
    pass
