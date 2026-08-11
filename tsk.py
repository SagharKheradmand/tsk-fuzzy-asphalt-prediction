import numpy as np
from dataclasses import dataclass
from fcm import fuzzy_c_means


def gaussian_mf(X, c, s):
    """
    Gaussian membership function.
    """
    s = np.where(s < 1e-6, 1e-6, s)
    z = (X - c) / s
    return np.exp(-0.5 * np.sum(z**2, axis=1))


@dataclass
class TSKModel:
    n_rules: int
    m: float
    max_iter: int
    tol: float
    ridge_lambda: float
    seed: int

    fcm_: object = None
    A_: np.ndarray = None

    def _compute_firing(self, X):
        R = self.n_rules
        N = X.shape[0]
        W = np.zeros((R, N))
        for r in range(R):
            W[r] = gaussian_mf(X, self.fcm_.centers[r], self.fcm_.sigmas[r])
        W /= np.sum(W, axis=0, keepdims=True) + 1e-12
        return W

    def fit(self, X, y):
        """
        Train TSK model.
        """
        self.fcm_ = fuzzy_c_means(
            X, self.n_rules, self.m, self.max_iter, self.tol, self.seed
        )

        Wn = self._compute_firing(X)

        n, d = X.shape
        Phi = np.hstack([np.ones((n, 1)), X])
        self.A_ = np.zeros((self.n_rules, d + 1))

        I = np.eye(d + 1)
        for r in range(self.n_rules):
            W = np.diag(Wn[r])
            left = Phi.T @ W @ Phi + self.ridge_lambda * I
            right = Phi.T @ W @ y
            self.A_[r] = np.linalg.solve(left, right)

        return self

    def predict(self, X):
        """
        Predict output values.
        """
        Phi = np.hstack([np.ones((X.shape[0], 1)), X])
        Wn = self._compute_firing(X)
        Yr = self.A_ @ Phi.T
        return np.sum(Wn * Yr, axis=0)
