import numpy as np
from Gaussian import Gaussian


def Make2DFilt(fmat, f, filttype):
    """
    Python translation of Make2DFilt.m

    Constructs a 2D spectral filter.

    Parameters
    ----------
    fmat : ndarray
        Frequency matrix (same shape as the FFT spectrum).
    f : array_like
        Transition frequencies defining the filter:
            - 'lowpass' : [flo, fhi]
            - 'highpass': [flo, fhi]
            - 'bandpass': [flo1, flo2, fhi1, fhi2]
    filttype : {"lowpass", "highpass", "bandpass"}
        Type of filter.

    Returns
    -------
    F : ndarray
        2D filter matrix, same shape as fmat.
    """
    fmat = np.asarray(fmat, dtype=np.float64)
    f = np.asarray(f, dtype=np.float64)

    if filttype == "lowpass":
        flo, fhi = float(f[0]), float(f[1])
        mu = flo
        sigma = abs(fhi - flo) / 3.0
        F = Gaussian(fmat, mu, sigma)
        # F(fmat < flo) = 1
        F = F.copy()
        F[fmat < flo] = 1.0

    elif filttype == "highpass":
        flo, fhi = float(f[0]), float(f[1])
        mu = fhi
        sigma = abs(fhi - flo) / 3.0
        F = Gaussian(fmat, mu, sigma)
        # F(fmat >= fhi) = 1
        F = F.copy()
        F[fmat >= fhi] = 1.0

    elif filttype == "bandpass":
        flo1, flo2, fhi1, fhi2 = map(float, f)
        sigmalo = abs(flo2 - flo1) / 3.0
        sigmahi = abs(fhi2 - fhi1) / 3.0
        mulo = flo2
        muhi = fhi1

        Flo = Gaussian(fmat, mulo, sigmalo)
        Fhi = Gaussian(fmat, muhi, sigmahi)

        # F = Flo.*(fmat<=mulo) + Fhi.*(fmat>=muhi) + 1*(fmat>mulo & fmat<muhi);
        F = np.zeros_like(fmat, dtype=np.float64)

        mask_lo = fmat <= mulo
        mask_hi = fmat >= muhi
        mask_mid = (fmat > mulo) & (fmat < muhi)

        F[mask_lo] = Flo[mask_lo]
        F[mask_hi] = Fhi[mask_hi]
        F[mask_mid] = 1.0

    else:
        raise ValueError(f"Unsupported filter type: {filttype}")

    return F
