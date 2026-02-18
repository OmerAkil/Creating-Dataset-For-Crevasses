from pathlib import Path

import numpy as np
import rasterio
from scipy.signal import convolve2d

from mat2gray import mat2gray
from gaborfilter import gaborfilter
from histCountCut import histCountCut
# from gaborfilter import gaborfilter  # you'll implement this from gaborfilter.m
# from histCountCut import histCountCut  # you'll implement this from histCountCut.m


def multidirection_gabor(
    image_file,
    width,
    hist_count_threshold,
    outputpath,
    elongation=1.0,
    filter_type="even",
):
    """
    Python translation of multidirection_gabor.m

    Parameters
    ----------
    image_file : str or Path
        Path to input GeoTIFF (e.g. output of spectralanalysis).
    width : float or int
        Small river width for Gabor filter (controls scale).
    hist_count_threshold : int
        Pixel count threshold used by histCountCut().
    outputpath : str or Path
        Directory where output GeoTIFF will be written.
    elongation : float, optional
        Elongation of the Gabor filter in the orientation direction
        with respect to its thickness (default: 1.0).
    filter_type : {"even", "odd"}, optional
        Type of Gabor filter (default: "even").

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

    # Output filename depends on filter_type
    if filter_type == "even":
        output_file = outputpath / f"{name}_gabor{ext}"
    else:
        output_file = outputpath / f"{name}_gabor_odd{ext}"

    output_file_name = str(output_file)
    print(f"process {name}")

    # If already processed, skip
    if output_file.is_file():
        return output_file_name

    # --- Read GeoTIFF + metadata (equivalent to geotiffinfo / GeoKeyDirectoryTag) ---
    with rasterio.open(image_file) as src:
        image = src.read(1)  # first band only
        transform = src.transform
        profile = src.profile.copy()
        src_tags = src.tags()       # root-level tags (GeoTIFF + others)
        src_band1_tags = src.tags(1)  # per-band tags for band 1
        src_crs = src.crs

    # Convert to single precision like MATLAB single()
    image = image.astype(np.float32, copy=False)
    l, w = image.shape

    # Angles from 0 to 165 degrees in 15-degree steps (12 directions)
    angles_deg = np.arange(0, 180, 15)  # 0, 15, ..., 165
    assert len(angles_deg) == 12

    # Initialize accumulator image
    image_gf = np.zeros((l, w), dtype=np.float32)

    for theta_deg in angles_deg:
        theta = np.deg2rad(theta_deg)  # angle of rotation in radians

        gamma = 2.0 * width
        lambd = 2.0 * width  # lambda = 1/f
        psi = 0.0            # phase offset
        sigma = gamma / (2.0 * np.sqrt(2.0 * np.log(2.0)))

        # gaborfilter: Python version of MATLAB gaborfilter.m
        gb_filt = gaborfilter(
            sigma,
            theta,
            lambd,
            psi,
            elongation,   # in MATLAB they call this elongation
            filter_type,
        )

        # 2D convolution, same size, zero-padding (conv2(...,'same'))
        g = convolve2d(image, gb_filt, mode="same", boundary="fill", fillvalue=0.0)

        # Accumulate max response across directions
        image_gf = np.maximum(image_gf, g.astype(np.float32))

    # Normalize to [0, 1] like MATLAB mat2gray, then to 0–255 uint8
    image_gf = mat2gray(image_gf)
    # MATLAB does: uint8(image_gf * 255) -> truncation, not rounding
    image_gf = (image_gf * 255.0).astype(np.uint8)

    # Histogram-based clipping
    image_gf = histCountCut(image_gf, hist_count_threshold)

    # --- Write output GeoTIFF, preserving georeferencing & tags ---
    profile.update(
        dtype="uint8",
        count=1,
        height=image_gf.shape[0],
        width=image_gf.shape[1],
        crs=src_crs,
        transform=transform,
    )

    with rasterio.open(output_file, "w", **profile) as dst:
        dst.write(image_gf, 1)

        # Copy root-level tags (rough equivalent of GeoKeyDirectoryTag)
        if src_tags:
            dst.update_tags(**src_tags)

        # Copy band-1 tags from source to band 1 in output
        if src_band1_tags:
            dst.update_tags(1, **src_band1_tags)

    return output_file_name