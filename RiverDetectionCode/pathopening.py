import numpy as np
import diplib as dip


def pathopening(image, length, mode=("constrained", "robust")):
    """
    Wrapper around DIPlib's PathOpening, mimicking MATLAB's pathopening.

    Parameters
    ----------
    image : np.ndarray
        2D grayscale input image.
    length : int
        Path length threshold.
    mode : tuple(str)
        ('constrained','robust') or ('unconstrained',) etc.

    Returns
    -------
    np.ndarray
        Resulting image after path opening.
    """
    # Ensure 2D float32 array
    arr = np.asarray(image, dtype=np.float32)
    img_dip = dip.Image(arr)

    # DIPlib expects a set[str] for `mode` in Python bindings
    mode_set = set(mode)

    # Some builds use 'length', others 'filterSize'. For path opening the
    # argument name is 'length' in recent diplib versions.
    out = dip.PathOpening(img_dip, length=int(length), mode=mode_set)

    return np.asarray(out)
