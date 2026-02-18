import numpy as np


def lsplane(X):
    """
    Python translation of lsplane.m

    Least-squares plane (orthogonal distance regression).

    Parameters
    ----------
    X : ndarray, shape (m, 3)
        Each row is [x, y, z].

    Returns
    -------
    x0 : ndarray, shape (3,)
        Centroid of the data (a point on the best-fit plane).
    a : ndarray, shape (3,)
        Direction cosines (unit normal vector) of the best-fit plane.
    d : ndarray, shape (m,)
        Residual distances along the normal (optional in MATLAB; always returned here).
    normd : float
        Norm of the residual vector.
    """
    X = np.asarray(X, dtype=np.float64)
    m, n = X.shape
    if n != 3:
        raise ValueError("X must be an (m, 3) array of [x, y, z].")
    if m < 3:
        raise ValueError("At least 3 data points required.")

    # Centroid: mean of points (3,)
    x0 = X.mean(axis=0)

    # Translated points
    A = X - x0  # shape (m, 3)

    # SVD of A (equivalent to [U,S,V] = svd(A,0) in MATLAB)
    U, S, Vt = np.linalg.svd(A, full_matrices=False)

    # Smallest singular value and corresponding right singular vector
    i = int(np.argmin(S))
    s = S[i]
    a = Vt.T[:, i]  # right singular vector (3,)

    # Residual distances along the normal direction
    d = U[:, i] * s
    normd = float(np.linalg.norm(d))

    return x0, a, d, normd
