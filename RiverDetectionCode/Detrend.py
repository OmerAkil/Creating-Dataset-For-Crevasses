import numpy as np
from lsplane import lsplane

def Detrend(M):
    M = np.asarray(M, dtype=np.float64)
    ny, nx = M.shape

    x = np.arange(1, nx + 1, dtype=np.float64)
    y = np.arange(1, ny + 1, dtype=np.float64)
    X, Y = np.meshgrid(x, y)

    points = np.column_stack([X.ravel(), Y.ravel(), M.ravel()])

    # lsplane returns: x0 (centroid), a (normal), d, normd
    centroid, cosines, _, _ = lsplane(points)

    centroid = np.asarray(centroid, dtype=np.float64).ravel()
    cosines = np.asarray(cosines, dtype=np.float64).ravel()

    a = -cosines[0] / cosines[2]
    b = -cosines[1] / cosines[2]
    c = centroid[2] + (cosines[0] * centroid[0] + cosines[1] * centroid[1]) / cosines[2]

    plane = a * X + b * Y + c
    D = M - plane
    return D
