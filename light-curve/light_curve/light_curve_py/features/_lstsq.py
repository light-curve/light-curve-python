import numpy as np


def least_squares(t, m, sigma):
    A = np.vstack([t, np.ones(len(t))]).T

    if sigma is not None:
        w = np.diag(1 / sigma)
        A = np.dot(w, A)
        m = np.dot(w, m.T)

    (slope, _), residuals, *_ = np.linalg.lstsq(A, m, rcond=None)
    return slope, residuals[0]
