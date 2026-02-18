from pathlib import Path

import numpy as np
import rasterio


def mask_border(
    images,
    img_dir,
    crev_dir,
    out_dir,
    buffer_pixels=5,
    target_epsg="EPSG:3413",
):
    """
    Python rewrite of MaskBorder.m

    For each image ID:
      - Read full WorldView image <ID>.tif
      - Build a data mask (1 = valid, 0 = no-data) with a 5-pixel buffer
        around the start/end of the image data region (both rows & columns)
      - Read binary fracture image Crevasse_<ID>_binary_10.tif
      - Remove the artificial border by multiplying with the mask
      - Save:
          FracMap_<ID>.tif  (cleaned binary fracture map)
          DataMask_<ID>.tif (0/1 mask of valid data)
    """

    img_dir = Path(img_dir)
    crev_dir = Path(crev_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for image_id in images:
        print(f"Processing: {image_id}")

        # --- 1. Read WorldView file ---
        img_path = img_dir / f"{image_id}.tif"
        with rasterio.open(img_path) as src:
            img = src.read(1)  # first band
            transform = src.transform
            height, width = img.shape

        # Start with all-ones mask
        index = np.ones_like(img, dtype=np.uint8)

        # --- 2. Top & bottom borders (row-wise pass) ---
        for m in range(height):
            row = img[m, :]
            zero_inds = np.where(row == 0)[0]  # 0-based indices of zeros

            if len(zero_inds) == 0:
                # No zeros in this row -> leave mask row as is
                continue

            if len(zero_inds) < width:
                # Look for a gap in the zeros (where data starts/ends)
                diffs = np.diff(zero_inds)
                splits = np.where(diffs > 1)[0]

                if len(splits) > 0:
                    s = splits[0]
                    left_last_zero = zero_inds[s]
                    right_first_zero = zero_inds[s + 1]

                    left_end = min(left_last_zero + buffer_pixels, width - 1)
                    right_start = max(right_first_zero - buffer_pixels, 0)

                    # Zero out from left border through small buffer inside data
                    index[m, : left_end + 1] = 0
                    # Zero out from buffer inside data to right border
                    index[m, right_start:] = 0
                else:
                    # All zeros are contiguous but not the entire row:
                    # treat as no valid data
                    index[m, :] = 0
            else:
                # Entire row is zeros -> no data
                index[m, :] = 0

        # --- 3. Left & right borders (column-wise pass) ---
        for m in range(width):
            col = img[:, m]
            zero_inds = np.where(col == 0)[0]  # 0-based indices of zeros

            if len(zero_inds) == 0:
                continue

            if len(zero_inds) < height:
                diffs = np.diff(zero_inds)
                splits = np.where(diffs > 1)[0]

                if len(splits) > 0:
                    s = splits[0]
                    top_last_zero = zero_inds[s]
                    bottom_first_zero = zero_inds[s + 1]

                    top_end = min(top_last_zero + buffer_pixels, height - 1)
                    bottom_start = max(bottom_first_zero - buffer_pixels, 0)

                    index[: top_end + 1, m] = 0
                    index[bottom_start:, m] = 0
                else:
                    index[:, m] = 0
            else:
                index[:, m] = 0

        # --- 4. Load binary fracture file ---
        crev_path = crev_dir / f"Crevasse_{image_id}_binary_10.tif"
        with rasterio.open(crev_path) as src_crev:
            crev_img = src_crev.read(1)

        # Make sure shapes are consistent
        h2, w2 = crev_img.shape
        h = min(height, h2)
        w = min(width, w2)

        clean_data = (crev_img[:h, :w] * index[:h, :w]).astype(np.uint8)
        data_mask = index[:h, :w].astype(np.uint8)

        # --- 5. Write outputs (GeoTIFF) ---
        frac_out = out_dir / f"FracMap_{image_id}.tif"
        mask_out = out_dir / f"DataMask_{image_id}.tif"

        profile = {
            "driver": "GTiff",
            "height": h,
            "width": w,
            "count": 1,
            "dtype": "uint8",
            "crs": target_epsg,
            "transform": transform,
        }

        # Clean binary fracture map
        with rasterio.open(frac_out, "w", **profile) as dst:
            dst.write(clean_data, 1)

        # Data mask
        with rasterio.open(mask_out, "w", **profile) as dst:
            dst.write(data_mask, 1)

        print(f"  Wrote: {frac_out.name}, {mask_out.name}")


# ============================
# Example usage (edit paths!)
# ============================

#images = [
#    "WV01_20120803164856", "QB02_20120729152314", "QB02_20120731154958",
#    "QB02_20120731155001", "QB02_20120731155004", "WV01_20120713164417",
#    "WV01_20120713164418", "WV01_20120713164419", "WV01_20120803164853",
#    "WV01_20120803164854", "WV01_20120803164855",
#    "WV01_20120802153817", "WV01_20120802153816", "WV01_20120802153815",
#    "WV01_20120802153814", "WV01_20120802153813", "WV01_20120713005153",
#    "WV01_20120713005152", "WV01_20120713005151",
#]

# Image ID list
images = ["tae_8_crevasse_gray"]

PROJECT_ROOT = Path(__file__).resolve().parent.parent

img_dir  = PROJECT_ROOT / "InputImages"                # where tae_8_crevasse_gray.tif lives
crev_dir = PROJECT_ROOT / "PostProcessing" / "Clipped" # where clipped Crevasse file is
out_dir  = PROJECT_ROOT / "PostProcessing" / "Masked"  # output folder (will be created)

mask_border(images, img_dir=img_dir, crev_dir=crev_dir, out_dir=out_dir)

