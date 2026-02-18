from pathlib import Path

from spectralanalysis import spectralanalysis
from multidirection_gabor import multidirection_gabor
from im_pathopening import im_pathopening


def run_batch_river_detection(
    input_dir,
    output_dir,
    sensor="WV",
    inverse=1,
    width=2,
    ppolength=20,
    smooth=0.7,
    histCountThreshold=100,
):
    """
    Python translation of run_batch_river_detection.m

    Parameters
    ----------
    input_dir : str or Path
        Directory containing input .tif images (e.g. WV / Sentinel2 scenes).
    output_dir : str or Path
        Directory where processed outputs will be written.
    sensor : str
        One of: 'WV', 'SPOT', 'SETSM', 'Sentinel2', 'Landsat', 'LandsatNDWI'.
    inverse : int
        1 = convert dark rivers to bright (e.g., panchromatic image).
        0 = keep bright rivers (e.g., NDWI images).
    width : int
        Small river width for Gabor filter.
    ppolength : int
        Path opening length.
    smooth : float
        Smooth parameter for denoise algorithm (not used in this script, but kept for completeness).
    histCountThreshold : int
        Pixel-count threshold for histogram-based stretching (e.g. 100 or 1000).
    """

    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(input_dir.glob("*.tif"))
    m = len(files)
    print(f"Found {m} .tif files in {input_dir}")

    # MATLAB: for i = 3:m  (1-based index, skipping first two files)
    # Python: indices 2 .. m-1 
    #for i in range(2, m)
    for i in range(m):
        image_path = files[i]
        image_name = image_path.name
        print(f"\nProcessing file {i+1}/{m}: {image_name} (sensor={sensor})")

        if sensor == "WV":
            f = [1 / 100, 1 / 20, 1 / 5, 1 / 1]
            filterType = "bandpass"

            imageFFT = spectralanalysis(
                image_file=image_path,
                frequency=f,
                filter_type=filterType,
                inverse=inverse,
                outputpath=output_dir,
            )
            imageGabor = multidirection_gabor(
                image_file=imageFFT,
                width=width,
                hist_count_threshold=histCountThreshold,
                outputpath=output_dir,
            )
            imagePPO = im_pathopening(
                image_file=imageGabor,
                length_threshold=ppolength,
                hist_count_threshold=histCountThreshold,
                outputpath=output_dir,
            )

        elif sensor == "SPOT":
            f = [1 / 200, 1 / 50, 1 / 10, 1 / 5]
            filterType = "bandpass"

            imageFFT = spectralanalysis(image_path, f, filterType, inverse, output_dir)
            imageGabor = multidirection_gabor(imageFFT, width, histCountThreshold, output_dir)
            imagePPO = im_pathopening(imageGabor, ppolength, histCountThreshold, output_dir)

        elif sensor == "SETSM":
            f = [1 / 200, 1 / 100, 1 / 20, 1 / 10]  # follows Karlstrom and Yang, 2016
            filterType = "bandpass"
            smooth = 0.5
            # Note: original MATLAB code only sets parameters for SETSM here;
            # actual processing would be added when you have the SETSM-specific scripts.

        elif sensor == "Sentinel2":
            f = [1 / 600, 1 / 200, 1 / 40, 1 / 20]
            filterType = "bandpass"

            imageFFT = spectralanalysis(image_path, f, filterType, inverse, output_dir)
            imageGabor = multidirection_gabor(imageFFT, width, histCountThreshold, output_dir)
            imagePPO = im_pathopening(imageGabor, ppolength, histCountThreshold, output_dir)

        elif sensor == "Landsat":
            f = [1 / 1500, 1 / 600, 1 / 200, 1 / 50]
            filterType = "bandpass"

            imageFFT = spectralanalysis(image_path, f, filterType, inverse, output_dir)
            imageGabor = multidirection_gabor(imageFFT, width, histCountThreshold, output_dir)
            imagePPO = im_pathopening(imageGabor, ppolength, histCountThreshold, output_dir)

        elif sensor == "LandsatNDWI":
            f = [1 / 1500, 1 / 600, 1 / 200, 1 / 50]
            filterType = "bandpass"

            imageFFT = spectralanalysis(image_path, f, filterType, inverse, output_dir)
            imageGabor = multidirection_gabor(imageFFT, width, histCountThreshold, output_dir)
            imagePPO = im_pathopening(imageGabor, ppolength, histCountThreshold, output_dir)

        else:
            print("  do not support this format:", sensor)

    print("\nBatch river detection finished.")
