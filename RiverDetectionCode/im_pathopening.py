from pathlib import Path

import numpy as np
import rasterio

from mat2gray import mat2gray
from histCountCut import histCountCut
from pathopening import pathopening

def im_pathopening(image_file, length_threshold, hist_count_threshold, outputpath):
    """
    Python translation of im_pathopening.m

    Parameters
    ----------
    image_file : str or Path
        Input GeoTIFF path (e.g. output of multidirection_gabor).
    length_threshold : int or float
        Path opening length (same as lengthThreshold in MATLAB).
    hist_count_threshold : int
        Threshold passed to histCountCut.
    outputpath : str or Path
        Directory where the path-opening result will be written.

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

    # --- MATLAB-like num2str for lengthThreshold ---
    # MATLAB num2str uses a %g-style formatting: no excessive decimals,
    # no trailing zeros, "3" instead of "3.0", etc.
    lt = float(length_threshold)
    length_str = f"{lt:g}"  # e.g. 3.0 -> "3", 3.75 -> "3.75", 0.0037 -> "0.0037"

    output_file = outputpath / f"{name}_cpo{length_str}{ext}"
    output_file_name = str(output_file)

    print(f"process {name}")

    # Skip if already exists (mimic MATLAB isfile check)
    if output_file.is_file():
        return output_file_name

    # --- Read GeoTIFF + metadata (like geotiffinfo/geotiffread) ---
    with rasterio.open(image_file) as src:
        image = src.read(1)  # first band
        transform = src.transform
        crs = src.crs
        profile = src.profile.copy()
        src_tags = src.tags()         # root-level tags (GeoTIFF + others)
        src_band1_tags = src.tags(1)  # per-band tags for band 1

    # Convert to single precision like MATLAB single()
    image = image.astype(np.float32, copy=False)

    # In MATLAB:
    #   dip_data = mat2im(image);
    #   out = pathopening(dip_data,lengthThreshold,{'constrained','robust'});
    #   image_path_opened = im2mat(out);
    #
    # Here we treat 'image' as our array and call a Python pathopening()
    image_path_opened = pathopening(
        image,
        length_threshold,
        mode=("constrained", "robust"),
    )

    # --- Post-processing (same as MATLAB) ---

    # Pixels smaller than mean cannot be rivers (for final display).
    meanPO = float(np.mean(image_path_opened))
    image_path_opened = image_path_opened.copy()
    image_path_opened[image_path_opened < meanPO] = meanPO

    # Normalize to [0,1] via mat2gray, then to 0–255 uint8
    image_path_opened = mat2gray(image_path_opened)
    # MATLAB: uint8(image_path_opened * 255) -> truncation, not rounding
    image_path_opened = (image_path_opened * 255.0).astype(np.uint8)

    # Apply histogram-based clipping/stretch
    image_path_opened = histCountCut(image_path_opened, hist_count_threshold)

    # --- Write GeoTIFF, preserving georeferencing & tags ---
    profile.update(
        dtype="uint8",
        count=1,
        height=image_path_opened.shape[0],
        width=image_path_opened.shape[1],
        transform=transform,
        crs=crs,
    )

    with rasterio.open(output_file, "w", **profile) as dst:
        dst.write(image_path_opened, 1)

        # Copy root-level tags (rough equivalent of GeoKeyDirectoryTag)
        if src_tags:
            dst.update_tags(**src_tags)

        # Copy band-1 tags from source to band 1 in output
        if src_band1_tags:
            dst.update_tags(1, **src_band1_tags)

    return output_file_name