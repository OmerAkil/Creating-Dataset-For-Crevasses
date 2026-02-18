import numpy as np
from fft2D import fft2D        
from Make2DFilt import Make2DFilt 


def SpecFilt2D(M, dx, dy, f, filttype):
    """
    Python translation of SpecFilt2D.m

    Filters a matrix M in the spectral domain.

    Parameters
    ----------
    M : 2D ndarray
        Input matrix (spatial domain).
    dx, dy : float
        Cell sizes in x and y directions.
    f : array_like
        Array of transition frequencies that define the 2D filter.
        - lowpass : [flo, fhi]
        - highpass: [flo, fhi]
        - bandpass: [flo1, flo2, fhi1, fhi2]
    filttype : {"lowpass", "highpass", "bandpass"}
        Type of filter.

    Returns
    -------
    M_filtered : 2D ndarray
        Filtered matrix (spatial domain).
    F : 2D ndarray
        Filter matrix in the spectral domain.
    """

    M = np.asarray(M, dtype=np.float64)
    ny, nx = M.shape  # number of rows and columns

    # ------------------------------------------------------------------
    # get frequency matrix using fft2D (like [Pmat, fmat] = fft2D(...))
    # ------------------------------------------------------------------
    pad = 0
    window = 0
    Pmat, fmat, _, _ = fft2D(M, dx, dy, pad, window)

    # NOTE (from MATLAB comments):
    # % Remove first-order (planar) trend, but do not window
    # % M = Detrend(M);
    # detrending is already handled in spectralanalysis.m, so we DO NOT
    # apply it again here.

    # ------------------------------------------------------------------
    # Do a 2D FFT, centered (fftshift)
    # ------------------------------------------------------------------
    M_fft = np.fft.fft2(M)
    M_fft_shift = np.fft.fftshift(M_fft)

    # ------------------------------------------------------------------
    # Make the filter matrix F in the spectral domain
    # ------------------------------------------------------------------
    F = Make2DFilt(fmat, f, filttype)

    # ------------------------------------------------------------------
    # Apply the filter and inverse FFT back to spatial domain
    # ------------------------------------------------------------------
    M_filtered_fft_shift = M_fft_shift * F
    M_filtered = np.fft.ifft2(np.fft.ifftshift(M_filtered_fft_shift))
    M_filtered = np.real(M_filtered)

    # Crop back (in case padding was used in fft2D)
    M_filtered = M_filtered[0:ny, 0:nx]

    return M_filtered, F
