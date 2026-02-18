import numpy as np


def gaborfilter(sigma, theta, lambd, psi, elongation, filter_type="even"):
    """
    Python translation of gaborfilter.m

    Parameters
    ----------
    sigma : float
        Base standard deviation of the Gaussian envelope.
    theta : float
        Orientation in radians.
    lambd : float
        Wavelength of the sinusoidal factor (lambda).
    psi : float
        Phase offset.
    elongation : float
        Elongation factor in the y-direction (sigma_y = sigma * elongation).
    filter_type : {"even", "odd"}, optional
        Type of Gabor filter:
            "even" -> cosine
            "odd"  -> sine

    Returns
    -------
    gb : 2D ndarray (float64)
        Gabor filter kernel.
    """

    sigma_x = float(sigma)
    sigma_y = float(sigma) * float(elongation)

    # Bounding box: nstds standard deviations
    nstds = 3.0

    xmax = max(
        abs(nstds * sigma_x * np.cos(theta)),
        abs(nstds * sigma_y * np.sin(theta)),
    )
    xmax = int(np.ceil(max(1.0, xmax)))

    ymax = max(
        abs(nstds * sigma_x * np.sin(theta)),
        abs(nstds * sigma_y * np.cos(theta)),
    )
    ymax = int(np.ceil(max(1.0, ymax)))

    xmin, ymin = -xmax, -ymax

    # Coordinate grid
    x, y = np.meshgrid(
        np.arange(xmin, xmax + 1),
        np.arange(ymin, ymax + 1),
    )

    # Rotation
    x_theta = x * np.cos(theta) + y * np.sin(theta)
    y_theta = -x * np.sin(theta) + y * np.cos(theta)

    # Gaussian envelope
    gauss = np.exp(
        -0.5 * ((x_theta ** 2) / (sigma_x ** 2) + (y_theta ** 2) / (sigma_y ** 2))
    )
    norm = 1.0 / (2.0 * np.pi * sigma_x * sigma_y)

    # Sinusoidal carrier
    if filter_type == "even":
        carrier = np.cos(2.0 * np.pi * x_theta / lambd + psi)
    else:
        carrier = np.sin(2.0 * np.pi * x_theta / lambd + psi)

    gb = norm * gauss * carrier
    return gb
