import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
from scipy import linalg


class KernelPCA:

    def kpca(self, X: np.ndarray, y: np.ndarray, Xt: np.ndarray, s: float = 0.1) -> tuple:

        n_samples, n_org_ftr = X.shape
        n_samplest, n_org_ftrt = Xt.shape
        e = np.ones((n_samples,), np.float).T
        L = np.zeros((n_samples, n_samples), np.int)
        H = np.identity(n_samples, np.float) - \
            ((n_samples ** -1) * (e @ e.T))
        K = np.zeros((n_samples, n_samples), np.float)

        for i in range(n_samples):
            L[i][y == y[i]] = 1
            K[i] = np.exp((-np.linalg.norm(X - X[i], axis=1) ** 2) / (2 * s ** 2))  # distance train and train
        Q = K @ H @ L @ H @ K

        vals, beta = linalg.eigh(Q, K)

        beta = beta[:, np.argsort(vals)[::-1]]

        K_test = np.zeros((X.shape[0], n_samplest), np.float)
        for i in range(n_samplest):
            K_test[:, i] = np.exp((-np.linalg.norm(X - Xt[i], axis=1) ** 2) / (2 * s ** 2))

        beta = beta[:, :2]

        return (beta.T @ K).T, (beta.T @ K_test).T  # projection
