from pathlib import Path
import numpy as np
import rasterio

def binarize_images(
    in_dir: str,
    out_dir: str,
    threshold: float = 10.0,
    input_suffix: str = "_bandpass_gabor_cpo20.tif",
    output_suffix: str = "_binary_10.tif",
):
    """
    Python rewrite of BinarizeImages.m

    - Finds all GeoTIFFs in `in_dir` whose names end with `input_suffix`
    - For each, reads the raster, thresholds it:
         pixel > threshold -> 1
         pixel <= threshold -> 0
    - Saves result to `out_dir` as uint8 GeoTIFF with same georeferencing,
      but filename suffix replaced by `output_suffix`.
    """
    in_path = Path(in_dir)
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    tif_files = sorted(in_path.glob(f"*{input_suffix}"))
    print(f"Found {len(tif_files)} input files in {in_path}")

    for tif_path in tif_files:
        print("Processing:", tif_path.name)

        # Read source raster
        with rasterio.open(tif_path) as src:
            # Assume single-band, but handle multi-band gracefully
            data = src.read(1)  # first band
            profile = src.profile.copy()

        # Threshold to binary (0/1)
        binary = (data > threshold).astype("uint8")

        # Update profile: uint8, 1 band
        profile.update(
            dtype="uint8",
            count=1
        )

        # Build output filename:
        #   e.g. foo_bandpass_gabor_cpo20.tif -> foo_binary_10.tif
        if tif_path.name.endswith(input_suffix):
            base_name = tif_path.name[: -len(input_suffix)]
        else:
            base_name = tif_path.stem  # fallback

        out_name = base_name + output_suffix
        out_file = out_path / out_name

        with rasterio.open(out_file, "w", **profile) as dst:
            dst.write(binary, 1)

    print("Done. Binary images written to:", out_path)


# === PATHS (relative to project root) ===
PROJECT_ROOT = Path(__file__).resolve().parent.parent
in_dir = PROJECT_ROOT / "RiverOutput"
out_dir = PROJECT_ROOT / "PostProcessing" / "BinaryOutput"

binarize_images(in_dir=in_dir, out_dir=out_dir, threshold=10.0)

