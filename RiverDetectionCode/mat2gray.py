import numpy as np


def mat2gray(A, limits=None):
    """
    MATLAB-like mat2gray.

    I = mat2gray(A, [amin, amax]) converts the matrix A to an intensity image I.
    The output I is float64 with values in [0.0, 1.0].

    If limits is None:
        amin, amax = min(A(:)), max(A(:))
    else:
        amin, amax = limits

    If amax == amin (constant image):
        I = double(A), then clipped to [0, 1].

    Parameters
    ----------
    A : array_like
        Input array (any numeric dtype or boolean).
    limits : tuple or list of 2 floats, optional
        [amin, amax] values in A that map to 0.0 and 1.0.

    Returns
    -------
    I : ndarray (float64)
        Intensity image with values in [0, 1].
    """
    A = np.asarray(A)

    # Convert to float64 (MATLAB's default double)
    A = A.astype(np.float64, copy=False)

    if limits is None:
        # Handle case where A might be all NaN or empty
        if A.size == 0:
            return np.zeros_like(A, dtype=np.float64)

        amin = float(np.nanmin(A))
        amax = float(np.nanmax(A))
    else:
        if len(limits) != 2:
            raise ValueError("limits must be a sequence of two values [amin, amax].")
        amin = float(limits[0])
        amax = float(limits[1])

    if amax == amin:
        # Constant image: MATLAB does I = double(A) and then clamps to [0, 1]
        I = A.copy()
    else:
        delta = 1.0 / (amax - amin)
        # MATLAB: imlincomb(delta, A, -amin*delta, 'double')
        I = delta * A - amin * delta

    # Clamp to [0, 1]
    I = np.maximum(0.0, np.minimum(I, 1.0))

    return I
