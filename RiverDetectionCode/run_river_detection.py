from pathlib import Path

from spectralanalysis import spectralanalysis
from multidirection_gabor import multidirection_gabor
from im_pathopening import im_pathopening


def run_river_detection(
    image_file,
    outputpath,
    sensor="Sentinel2",
    inverse=0,
    width=2,
    ppolength=20,
    smooth=0.7,
    histCountThreshold=1000,
):
    """
    Python translation of run_river_detection.m

    Parameters
    ----------
    image_file : str or Path
        Input GeoTIFF (e.g. test_sentinel2_image.tif).
    outputpath : str or Path
        Directory where outputs will be written.
    sensor : str
        One of: 'WV', 'SPOT', 'SETSM', 'Sentinel2', 'Landsat', 'LandsatNDWI'.
    inverse : int
        1 = convert dark rivers to bright (e.g., panchromatic).
        0 = keep bright rivers (e.g., NDWI images).
    width : int
        Small river width for Gabor filter.
    ppolength : int
        Path opening length.
    smooth : float
        Smooth parameter for denoise algorithm (not used here, kept for completeness).
    histCountThreshold : int
        Pixel-count threshold for histogram-based stretching (e.g. 1000).

    Returns
    -------
    imageFFT, imageGabor, imagePPO : str
        Paths to intermediate and final products.
    """
    image_file = Path(image_file)
    outputpath = Path(outputpath)
    outputpath.mkdir(parents=True, exist_ok=True)

    print(f"Running river detection on {image_file.name} (sensor={sensor})")

    if sensor == "WV":          # 0.5 m resolution
        f = [1 / 100, 1 / 20, 1 / 5, 1 / 1]
        filterType = "bandpass"

    elif sensor == "SPOT":      # 1.5 m resolution
        f = [1 / 200, 1 / 50, 1 / 10, 1 / 5]
        filterType = "bandpass"

    elif sensor == "SETSM":     # 2.0 m resolution
        f = [1 / 200, 1 / 100, 1 / 20, 1 / 10]  # Karlstrom & Yang, 2016
        filterType = "bandpass"

    elif sensor == "Sentinel2": # 10 m resolution
        f = [1 / 600, 1 / 200, 1 / 40, 1 / 20]
        filterType = "bandpass"

    elif sensor == "Landsat":   # 15 m resolution
        f = [1 / 1000, 1 / 500, 1 / 200, 1 / 50]
        filterType = "bandpass"

    elif sensor == "LandsatNDWI":  # 30 m resolution
        f = [1 / 1000, 1 / 500, 1 / 200, 1 / 50]
        filterType = "bandpass"

    else:
        raise ValueError(f"Sensor type not supported: {sensor}")

    # --- Pipeline: FFT → Gabor → path opening ---

    imageFFT = spectralanalysis(
        image_file=image_file,
        frequency=f,
        filter_type=filterType,
        inverse=inverse,
        outputpath=outputpath,
    )

    imageGabor = multidirection_gabor(
        image_file=imageFFT,
        width=width,
        hist_count_threshold=histCountThreshold,
        outputpath=outputpath,
    )

    imagePPO = im_pathopening(
        image_file=imageGabor,
        length_threshold=ppolength,
        hist_count_threshold=histCountThreshold,
        outputpath=outputpath,
    )

    print("River detection finished.")
    return imageFFT, imageGabor, imagePPO
