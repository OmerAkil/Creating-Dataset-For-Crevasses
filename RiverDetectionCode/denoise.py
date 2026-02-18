from pathlib import Path

import numpy as np
import rasterio

from mat2gray import mat2gray  # your own MATLAB-style mat2gray
from rgb2gray import rgb2gray  # your MATLAB-like rgb2gray


def denoise(image_file, smooth_parameter, outputpath):
    """
    Python translation of denoise.m

    Parameters
    ----------
    image_file : str or Path
        Input GeoTIFF path.
    smooth_parameter : float
        Smoothing parameter h for bnlm2D.
        (e.g. 0.7 for WV/SPOT, 0.5 for SETSM).
    outputpath : str or Path
        Directory where the normalized + denoised image will be saved.

    Returns
    -------
    output_file_name : str
        Path to the output GeoTIFF.
    """

    image_file = Path(image_file)
    outputpath = Path(outputpath)
    outputpath.mkdir(parents=True, exist_ok=True)

    name = image_file.stem
    ext = image_file.suffix

    print(f"process {name}")

    # Parameters from MATLAB code
    M = 7           # search area size (2*M + 1)^2
    alpha = 3       # patch size (2*alpha + 1)^2
    h = smooth_parameter
    offset = 100.0  # to avoid NaN / div-by-zero in Pearson divergence

    # --- Read GeoTIFF + metadata (like geotiffinfo + geotiffread) ---
    with rasterio.open(image_file) as src:
        # Read all bands; we'll decide if it's RGB or single band
        data = src.read()             # shape: (bands, H, W)
        transform = src.transform
        profile = src.profile.copy()
        src_tags = src.tags()         # root-level tags (GeoTIFF + others)
        src_band1_tags = src.tags(1)  # per-band tags for band 1
        src_crs = src.crs

    # Determine if we have multi-band or single-band
    # data shape: (bands, height, width)
    if data.ndim == 3 and data.shape[0] > 1:
        # Convert to grayscale (supports (3,H,W) etc.)
        img = rgb2gray(data)
    else:
        img = data[0].astype(np.float64)

    # Intensity normalization (similar to MATLAB)
    imgd = img.astype(np.float64)
    mini = np.min(imgd)
    imgd = imgd - mini
    maxi = np.max(imgd)

    if maxi > 0:
        imgd = imgd / maxi * 255.0
    else:
        # Constant image: avoid division by zero
        imgd = np.zeros_like(imgd)

    # Add offset to enable Pearson divergence computation (avoid /0)
    imgd = imgd + offset
    s = imgd.shape  # (rows, cols)

    # --- Symmetric padding (padarray(...,'symmetric')) ---
    pad_width = ((alpha, alpha), (alpha, alpha))
    imgd_padded = np.pad(imgd, pad_width, mode="symmetric")

    # --- Call non-local means-based denoising: bnlm2D ---
    # You must implement this from bnlm2D.m
    fimgd_padded = bnlm2D(imgd_padded, M, alpha, h)

    # Remove offset
    fimgd_padded = fimgd_padded - offset
    imgd_padded = imgd_padded - offset

    # Crop back to original size
    r, c = s
    fimgd = fimgd_padded[alpha:alpha + r, alpha:alpha + c]
    imgd_original = imgd_padded[alpha:alpha + r, alpha:alpha + c]

    # Normalize filtered image by max of original: fimg = fimgd / max(imgd(:))
    max_orig = np.max(imgd_original) if imgd_original.size > 0 else 1.0
    if max_orig > 0:
        fimg = fimgd / max_orig
    else:
        fimg = np.zeros_like(fimgd)

    # Optional inversion (commented out in MATLAB):
    # fimg = fimg.max() - fimg

    # Convert to [0,1] via mat2gray, then to uint8 (no rounding, MATLAB-style truncation)
    fimg = mat2gray(fimg)
    fimg_uint8 = (fimg * 255.0).astype("uint8")

    # --- Write output GeoTIFF, preserving georeferencing & tags ---
    output_file = outputpath / f"{name}_norm_denoised{ext}"
    output_file_name = str(output_file)

    profile.update(
        dtype="uint8",
        count=1,
        height=fimg_uint8.shape[0],
        width=fimg_uint8.shape[1],
        crs=src_crs,
        transform=transform,
    )

    with rasterio.open(output_file, "w", **profile) as dst:
        dst.write(fimg_uint8, 1)

        # Copy root-level tags (rough equivalent of GeoKeyDirectoryTag)
        if src_tags:
            dst.update_tags(**src_tags)

        # Copy band-1 tags from source to band 1 in output
        if src_band1_tags:
            dst.update_tags(1, **src_band1_tags)

    return output_file_name


# ----------------------------------------------------------------------
# Placeholder for external function you will implement separately
# ----------------------------------------------------------------------

def bnlm2D(imgd_padded, M, alpha, h):
    """
    Placeholder for MATLAB bnlm2D.m (bilateral/non-local means 2D).

    Implement this using the original MATLAB bnlm2D.m logic.

    Parameters
    ----------
    imgd_padded : ndarray
        Padded input image (float).
    M : int
        Search area parameter; window size is (2*M+1)^2.
    alpha : int
        Patch size parameter; patch is (2*alpha+1)^2.
    h : float
        Smoothing parameter.

    Returns
    -------
    fimgd_padded : ndarray
        Denoised image, same shape as imgd_padded.
    """
    raise NotImplementedError("bnlm2D() must be implemented from bnlm2D.m.")
