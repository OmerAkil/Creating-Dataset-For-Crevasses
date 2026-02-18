from pathlib import Path

import numpy as np
import rasterio
from mat2gray import mat2gray
from Detrend import Detrend
from SpecFilt2D import SpecFilt2D

def spectralanalysis(image_file, frequency, filter_type, inverse, outputpath):
    """
    Python translation of spectralanalysis.m

    Parameters
    ----------
    image_file : str or Path
        Path to input GeoTIFF.
    frequency : list or array of float
        Frequency band parameters, e.g. [1/100, 1/20, 1/5, 1/1].
    filter_type : str
        Type of spectral filter, e.g. 'bandpass'.
    inverse : int
        If 1, invert dark rivers -> bright (255 - value).
        If 0, keep bright rivers as-is.
    outputpath : str or Path
        Directory where output GeoTIFF will be written.

    Returns
    -------
    output_file_name : str
        Path to the output GeoTIFF file.
    """

    image_file = Path(image_file)
    outputpath = Path(outputpath)
    outputpath.mkdir(parents=True, exist_ok=True)

    name = image_file.stem
    ext = image_file.suffix
    output_file = outputpath / f"{name}_{filter_type}{ext}"
    output_file_name = str(output_file)

    print(f"process {name}")

    # If output already exists, do nothing (mimic MATLAB isfile check)
    if output_file.is_file():
        return output_file_name

    # --- Read GeoTIFF ---
    with rasterio.open(image_file) as src:
        Z = src.read(1)  # first band
        transform = src.transform
        profile = src.profile.copy()

    # Use transform to approximate pixel size (similar to info.PixelScale)
    dx = abs(transform.a)
    dy = abs(transform.e)

    # Replace NaN / Inf / out-of-range with minValue (0)
    min_value = 0.0
    Z = Z.astype("float32")
    # finite mask
    finite_mask = np.isfinite(Z)
    Z[~finite_mask] = min_value

    # NDWI-like clipping and 16-bit WV NaN sentinel
    Z[Z < -1.0] = min_value   # for NDWI images
    Z[Z > 10000.0] = min_value  # 65535 etc. treated as no-data

    # Normalize to [0, 1] like MATLAB mat2gray
    Z = mat2gray(Z)

    # Original dimensions
    Ny, Nx = Z.shape
    orig_Ny, orig_Nx = Ny, Nx

    # fft2D cannot process odd numbers in their implementation,
    # so they pad to even dimensions.
    signY = 1
    signX = 1

    if Ny % 2 == 1:
        new_row = Z[Ny - 1, :][None, :]
        Z = np.vstack([Z, new_row])
        Ny += 1
        signY = 0

    if Nx % 2 == 1:
        new_col = Z[:, Nx - 1][:, None]
        Z = np.hstack([Z, new_col])
        Nx += 1
        signX = 0

    # --- Detrend (MATLAB: Z = Detrend(Z);) ---
    # NOTE: you must implement detrend_2d() separately,
    # translating Detrend.m into Python.
    Z = Detrend(Z)

    # --- Spectral filter (MATLAB: Zhp = SpecFilt2D(...);) ---
    # NOTE: implement spec_filt_2d() separately,
    # translating SpecFilt2D.m into Python.
    #Zhp = SpecFilt2D(Z, dx=dx, dy=dy, frequency=frequency, filter_type=filter_type)
    Zhp, _ = SpecFilt2D(Z, dx, dy, frequency, filter_type)

    # Remove the padded row/column if we added them
    if signY == 0:
        Zhp = Zhp[: Ny - 1, :]
    if signX == 0:
        Zhp = Zhp[:, : Nx - 1]

    # At this point, Zhp should be back to original dimensions
    # (Ny-1/Nx-1 after crop ≈ orig_Ny/orig_Nx)
    Zhp = Zhp[:orig_Ny, :orig_Nx]

    # Convert to [0, 1], then to uint8 0–255 (MATLAB: mat2gray, uint8*255)
    Zhp = mat2gray(Zhp)
    Zhp_uint8 = (Zhp * 255.0).astype("uint8")

    # invert dark rivers to bright rivers (if inverse == 1)
    if inverse == 1:
        Zhp_uint8 = 255 - Zhp_uint8

    # --- Write GeoTIFF, preserving georeferencing ---
    profile.update(
        dtype="uint8",
        count=1,
        height=Zhp_uint8.shape[0],
        width=Zhp_uint8.shape[1],
    )

    with rasterio.open(output_file, "w", **profile) as dst:
        dst.write(Zhp_uint8, 1)

    return output_file_name
