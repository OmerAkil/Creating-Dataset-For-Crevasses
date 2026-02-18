import numpy as np


def Hann2D(M):
    """
    Python translation of Hann2D.m

    Windows matrix M with an elliptical Hann (raised cosine) window.
    Returns:
        H   : windowed data
        Wss : summed square of weighting coefficients
    """
    M = np.asarray(M, dtype=np.float64)
    ny, nx = M.shape

    # Matrix coordinates of centroid of M (1-based, like MATLAB)
    a = (nx + 1) / 2.0
    b = (ny + 1) / 2.0

    # X, Y from 1..nx, 1..ny
    x = np.arange(1, nx + 1, dtype=np.float64)
    y = np.arange(1, ny + 1, dtype=np.float64)
    X, Y = np.meshgrid(x, y)  # shape (ny, nx)

    # Angular polar coordinate theta
    # theta = (X==a)*(pi/2) + (X~=a)*atan2((Y-b),(X-a));
    theta = np.where(
        X == a,
        np.pi / 2.0,
        np.arctan2(Y - b, X - a),
    )

    # Radial coordinate
    r = np.sqrt((Y - b) ** 2 + (X - a) ** 2)

    # 'Radius' of ellipse for this theta
    # rprime = sqrt((a^2*b^2) * (b^2*cos^2(theta) + a^2*sin^2(theta))^(-1));
    denom = (b ** 2) * (np.cos(theta) ** 2) + (a ** 2) * (np.sin(theta) ** 2)
    rprime = np.sqrt((a ** 2 * b ** 2) / denom)

    # Hann coefficients: (r < rprime) * 0.5*(1 + cos(pi*r./rprime))
    hanncoeff = np.zeros_like(r, dtype=np.float64)
    mask = r < rprime
    hanncoeff[mask] = 0.5 * (1.0 + np.cos(np.pi * r[mask] / rprime[mask]))

    # Windowed data
    H = M * hanncoeff

    # Summed square of weighting coefficients
    Wss = np.sum(hanncoeff ** 2)

    return H, Wss
