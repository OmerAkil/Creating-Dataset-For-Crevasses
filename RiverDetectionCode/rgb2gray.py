import numpy as np


def rgb2gray(X):
    """
    MATLAB-like rgb2gray.

    Supports:
      - 3D RGB images:  (M, N, 3)
      - 3D bands-first: (3, M, N)  (we auto-convert to (M, N, 3))
      - 2D colormaps:   (K, 3)     (returns Kx3 grayscale colormap, like MATLAB)

    Dtypes:
      - uint8, uint16, single/float32, double/float64

    For 3D RGB:
      - float input -> output in same float dtype, clamped to [0, 1]
      - uint8/uint16 -> weighted sum, then clamped to [0, max] and cast back

    For 2D colormap:
      - expects float (double/single), returns float in [0,1], shape (K,3)
    """
    X = np.asarray(X)

    if X.ndim == 2:
        # 2D colormap: size(X,2) must be 3
        if X.shape[1] != 3 or X.shape[0] < 1:
            raise ValueError("Invalid size for colormap in rgb2gray: expected (K, 3).")
        if not np.issubdtype(X.dtype, np.floating):
            raise TypeError("For 2D colormap, X must be double/single (float).")

        # MATLAB: T = inv([...]); coef = T(1,:); but they effectively use [0.2989, 0.5870, 0.1140]
        coef = np.array([0.2989, 0.5870, 0.1140], dtype=np.float64)

        I1 = X @ coef  # (K, 3) @ (3,) -> (K,)
        I2 = np.clip(I1, 0.0, 1.0)

        # MATLAB: I = [I2,I2,I2]; horizontal concatenation
        I = np.stack([I2, I2, I2], axis=1)  # (K, 3)
        return I.astype(X.dtype, copy=False)

    elif X.ndim == 3:
        # 3D RGB: either (M, N, 3) or (3, M, N)
        if X.shape[2] == 3:
            img = X
        elif X.shape[0] == 3:
            # bands-first (3, H, W) -> (H, W, 3)
            img = np.moveaxis(X, 0, -1)
        else:
            raise ValueError("Invalid RGB image size in rgb2gray: expected (M,N,3) or (3,M,N).")

        if img.shape[2] != 3:
            raise ValueError("Invalid input size for RGB image in rgb2gray: 3 channels required.")

        dtype = img.dtype

        # Coefficients as in MATLAB doc (from their T matrix)
        coef = np.array([0.2989, 0.5870, 0.1140], dtype=np.float64)

        if np.issubdtype(dtype, np.floating):
            # double or single
            # Compute luminance, clamp to [0,1], same float dtype
            img_f = img.astype(np.float64, copy=False)
            Y = img_f @ coef  # (M,N,3) @ (3,) -> (M,N)
            Y = np.clip(Y, 0.0, 1.0)
            return Y.astype(dtype, copy=False)

        elif dtype == np.uint8 or dtype == np.uint16:
            # uint8 or uint16: multiply in double, then cast back, with clamp
            img_f = img.astype(np.float64)
            Y = img_f @ coef  # (M,N)
            if dtype == np.uint8:
                Y = np.clip(Y, 0.0, 255.0)
                return Y.astype(np.uint8)
            else:  # uint16
                Y = np.clip(Y, 0.0, 65535.0)
                return Y.astype(np.uint16)

        else:
            raise TypeError("rgb2gray only supports uint8, uint16, single, or double types for RGB.")

    else:
        raise ValueError("rgb2gray: invalid input size; expected 2D colormap or 3D RGB image.")
