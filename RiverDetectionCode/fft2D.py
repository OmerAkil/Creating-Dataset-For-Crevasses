import numpy as np
from Hann2D import Hann2D


def fft2D(M, dx, dy=None, pad=0, window=0):
    """
    Python translation of fft2D.m

    Computes the 2D power spectrum of the 2D matrix M.

    Parameters
    ----------
    M : 2D ndarray
        Data matrix (e.g., elevations).
    dx : float
        Spacing between matrix elements in x direction.
    dy : float, optional
        Spacing between matrix elements in y direction. If None, dy = dx.
    pad : int {0,1}
        If 1, pad with zeros to a power of 2. If 0, do not pad.
    window : int {0,1}
        If 1, apply elliptical Hann window. If 0, do not window.

    Returns
    -------
    Pm : 2D ndarray
        Matrix of spectral power (DFT periodogram).
    fm : 2D ndarray
        Matrix of radial frequency, units of 1/cellsize.
    Pv : 1D ndarray
        Vector of spectral power, sorted by increasing frequency.
    fv : 1D ndarray
        Vector of radial frequency, sorted by increasing frequency.
    """
    M = np.asarray(M, dtype=np.float64)

    if dy is None:
        dy = dx

    ny, nx = M.shape  # number of rows and columns

    # If either dimension is odd and no padding, bail out
    if not pad and ((nx % 2) or (ny % 2)):
        raise ValueError(
            "If either dimension of the input matrix is odd, "
            "it is recommended to pad with zeros (pad=1)."
        )

    # --- Data windowing ---
    if window:
        # Window with elliptical Hann window
        M, Wss = Hann2D(M)
    else:
        # No windowing (square window with value 1)
        Wss = float(ny * nx)

    # --- Data padding ---
    if pad:
        # Power-of-2 padding size
        L = int(2 ** np.ceil(np.log2(max(nx, ny))))
        Lx = L
        Ly = L
    else:
        Lx = nx
        Ly = ny

    # --- Frequency increments ---
    dfx = 1.0 / (dx * Lx)
    dfy = 1.0 / (dy * Ly)

    # --- 2D FFT with fftshift ---
    M_fft_shift = np.fft.fftshift(np.fft.fft2(M, s=(Ly, Lx)))

    # Zero out DC component at center
    yc0 = Ly // 2
    xc0 = Lx // 2
    M_fft_shift[yc0, xc0] = 0.0

    # DFT periodogram: |M|^2 / (Lx * Ly * Wss)
    M_power = M_fft_shift * np.conjugate(M_fft_shift)
    M_power = M_power.real / (Lx * Ly * Wss)

    # Assign power matrix
    Pm = M_power

    # --- Matrix of radial frequencies ---
    # Indices from 1..Lx, 1..Ly (like MATLAB)
    x = np.arange(1, Lx + 1, dtype=np.float64)
    y = np.arange(1, Ly + 1, dtype=np.float64)
    cols, rows = np.meshgrid(x, y)

    xc = Lx / 2.0 + 1.0
    yc = Ly / 2.0 + 1.0

    fm = np.sqrt((dfy * (rows - yc)) ** 2 + (dfx * (cols - xc)) ** 2)

    # --- Create sorted, non-redundant vectors Pv, fv ---

    # Take only first half in x: 1 .. (Lx/2+1)
    half_x = Lx // 2 + 1
    M_sub = M_power[:, :half_x]
    fv_sub = fm[:, :half_x].copy()

    # Mark redundant half-column with negative freq:
    # fv((yc+1):Ly, xc) = -1 in MATLAB.
    # yc index (1-based) => yc0 = Ly/2 (0-based), so rows yc0+1: end
    fv_sub[yc0 + 1 :, xc0] = -1.0

    # Flatten and sort by frequency
    freq_flat = fv_sub.ravel()
    power_flat = M_sub.ravel()

    # Stack as [freq, power] and sort by freq
    fp = np.column_stack([freq_flat, power_flat])
    fp_sorted = fp[np.argsort(fp[:, 0])]

    # Keep only positive frequencies (freq > 0)
    fp_pos = fp_sorted[fp_sorted[:, 0] > 0.0]

    # Separate into vectors
    fv = fp_pos[:, 0]
    Pv = 2.0 * fp_pos[:, 1]  # factor of 2 for taking only half of spectrum

    return Pm, fm, Pv, fv
