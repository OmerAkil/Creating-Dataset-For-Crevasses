import numpy as np


def Gaussian(freqmat, mu, sigma):
    """
    Python translation of Gaussian.m

    Parameters
    ----------
    freqmat : ndarray
        Frequency matrix (same shape as output filter).
    mu : float
        Center frequency.
    sigma : float
        Standard deviation of the Gaussian.

    Returns
    -------
    G : ndarray (float64)
        Normalized Gaussian with max(G) = 1.
    """
    freqmat = np.asarray(freqmat, dtype=np.float64)
    mu = float(mu)
    sigma = float(sigma)

    # Avoid division by zero
    if sigma == 0:
        raise ValueError("sigma must be non-zero in Gaussian()")

    G = np.exp(-((freqmat - mu) ** 2) / (2.0 * sigma ** 2))

    maxG = np.max(G)
    if maxG > 0:
        G = G / maxG

    return G
