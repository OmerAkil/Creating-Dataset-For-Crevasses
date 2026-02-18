import math
from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import xy
import xarray as xr
import pandas as pd


def create_image_chips(
    images,
    img_dir,
    crev_dir,
    kernel=None,
    valid_mask_threshold=0.75,
    min_fracture_density=1e-6,
):
    """
    Python rewrite of CreateImageChips.m

    Parameters
    ----------
    images : list of str
        Image IDs, e.g. ["WV01_20120802153817", ...]
    img_dir : str or Path
        Directory containing full WorldView images: <ID>.tif
    crev_dir : str or Path
        Directory containing:
            - FracMap_<ID>.tif
            - DataMask_<ID>.tif
        Also where .nc and CSV outputs will be written.
    kernel : int, optional
        Chip size in pixels. If None, uses floor(150 / 0.51).
    valid_mask_threshold : float
        Minimum mean of mask chip (0–1) required to keep chip.
    """

    img_dir = Path(img_dir)
    crev_dir = Path(crev_dir)
    crev_dir.mkdir(parents=True, exist_ok=True)

    # MATLAB: kernel = floor(150/0.51)
    if kernel is None:
        kernel = math.floor(150.0 / 0.51)
    print(f"Using kernel size: {kernel} x {kernel} pixels")

    total_tiles = 0

    for image_id in images:
        print(f"\nProcessing: {image_id}")

        # --- 1. Load binary fracture map ---
        frac_path = crev_dir / f"FracMap_{image_id}.tif"
        with rasterio.open(frac_path) as src_frac:
            frac_img = src_frac.read(1)          # binary fracture map
            transform = src_frac.transform       # affine transform
            crs = src_frac.crs
            height, width = frac_img.shape

        # --- 2. Load data mask ---
        mask_path = crev_dir / f"DataMask_{image_id}.tif"
        with rasterio.open(mask_path) as src_mask:
            mask_img = src_mask.read(1)          # mask with valid (1) / invalid (0)
            # assume same shape & transform as frac_img

        # --- 3. Load WorldView image ---
        wv_path = img_dir / f"{image_id}.tif"
        with rasterio.open(wv_path) as src_wv:
            # assume single-band grayscale or panchromatic
            wv_img = src_wv.read(1).astype(np.float32)  # 0–1 float geliyor

        # 0–1 aralığındaki görüntüyü 0–255 aralığına çek ve uint8 yap
        wv_img = (wv_img * 255.0).clip(0, 255).astype(np.uint8)

        print(f"{image_id}: WV img min/max after scale = {wv_img.min()} / {wv_img.max()}")


        y_dim, x_dim = frac_img.shape

        # leftover pixels in each dimension that don't make up a full chip
        x_del = x_dim % kernel
        y_del = y_dim % kernel

        # split leftovers in half: skip half on each side
        x_shift = math.ceil(x_del / 2)
        y_shift = math.ceil(y_del / 2)

        # number of chips in each direction
        ny = y_dim // kernel
        nx = x_dim // kernel

        # Lists for metadata
        x_start_list = []
        x_stop_list = []
        y_start_list = []
        y_stop_list = []
        x_coord_list = []
        y_coord_list = []

        # --- 4. Tile the image ---
        for m in range(ny):
            for p in range(nx):
                # NOTE: Python uses 0-based indexing and [start:stop) slicing.
                # We define chip bounds in 0-based pixel indices:
                c = m * kernel + y_shift           # row start (inclusive)
                d = (m + 1) * kernel + y_shift     # row stop (exclusive)
                a = p * kernel + x_shift           # col start (inclusive)
                b = (p + 1) * kernel + x_shift     # col stop (exclusive)

                # Safety check: stay within bounds (just in case)
                if d > y_dim or b > x_dim:
                    continue

                # Extract mask chip
                seg_mask = mask_img[c:d, a:b].astype(float)

                # Keep chip only if sufficiently valid
                if seg_mask.size == 0:
                    continue
                if seg_mask.mean() < valid_mask_threshold:
                    continue

                # Compute center pixel (0-based indices)
                x_center = int(math.floor(a + 0.5 * (b - a)))
                y_center = int(math.floor(c + 0.5 * (d - c)))

                # Real-world coordinates of center (rasterio: row, col)
                x_coord, y_coord = xy(transform, y_center, x_center, offset="center")

                # Store 1-based pixel indices for compatibility with MATLAB metadata
                x_start_list.append(a + 1)
                x_stop_list.append(b)
                y_start_list.append(c + 1)
                y_stop_list.append(d)
                x_coord_list.append(x_coord)
                y_coord_list.append(y_coord)

        # Convert lists to arrays
        x_start = np.array(x_start_list, dtype=np.int32)
        x_stop = np.array(x_stop_list, dtype=np.int32)
        y_start = np.array(y_start_list, dtype=np.int32)
        y_stop = np.array(y_stop_list, dtype=np.int32)
        x_coord = np.array(x_coord_list, dtype=np.float64)
        y_coord = np.array(y_coord_list, dtype=np.float64)

        # Generate IDs
        n_tiles = x_start.shape[0]
        if n_tiles == 0:
            print(f"  No valid tiles for {image_id}, skipping.")
            continue

        tile_id = np.arange(1, n_tiles + 1, dtype=np.int32)

        print(f"  Valid tiles: {n_tiles}")

        # --- 5. Extract image chips & label chips + fracture density ---
        keep_idx = []
        image_list = []
        label_list = []
        fd_list = []

        print(f"{image_id}: wv_img global min/max = {wv_img.min()} / {wv_img.max()}")

        for i in range(n_tiles):
            # Convert back to 0-based for slicing
            ys0 = y_start[i] - 1
            ye0 = y_stop[i]
            xs0 = x_start[i] - 1
            xe0 = x_stop[i]

            chip = wv_img[ys0:ye0, xs0:xe0]
            seg = frac_img[ys0:ye0, xs0:xe0] # Binary fracture map chip

            fd = float(seg.mean())

            # FILTER: drop tiles with no fractures
            if fd <= min_fracture_density:
                continue

            keep_idx.append(i)
            image_list.append(chip)
            label_list.append(seg)
            fd_list.append(fd)

        if len(keep_idx) == 0:
            print(f"  No tiles left after fracture filter for {image_id}, skipping.")
            continue

        # Subset metadata arrays to match kept tiles
        keep_idx = np.array(keep_idx, dtype=np.int32)
    
        x_start = x_start[keep_idx]
        x_stop  = x_stop[keep_idx]
        y_start = y_start[keep_idx]
        y_stop  = y_stop[keep_idx]
        x_coord = x_coord[keep_idx]
        y_coord = y_coord[keep_idx]  
            
        # Convert chip lists to arrays
        # Use int16 to avoid netCDF3 uint8 issues
        image_data = np.stack(image_list).astype(np.int16)
        label_data = np.stack(label_list).astype(np.int16)
        fracture_density = np.array(fd_list, dtype=np.float32)

        # Rebuild tile ids 1..N after filtering
        n_tiles = image_data.shape[0]
        tile_id = np.arange(1, n_tiles + 1, dtype=np.int32)
    
        print(f"  Valid tiles after fracture filter: {n_tiles}")

        # --- 6. Create NetCDF file via xarray ---
        nc_path = crev_dir / f"{image_id}.nc"

        ds = xr.Dataset(
            data_vars=dict(
                id=("tile", tile_id),
                x=("tile", x_coord),
                y=("tile", y_coord),
                x_start=("tile", x_start),
                x_stop=("tile", x_stop),
                y_start=("tile", y_start),
                y_stop=("tile", y_stop),
                fracture_density=("tile", fracture_density),
                image_data=(("tile", "y", "x"), image_data),
                label_data=(("tile", "y", "x"), label_data),
            ),
            coords=dict(
                tile=("tile", tile_id),
            ),
            attrs=dict(
                source_image=image_id,
                crs=str(crs) if crs is not None else "",
                kernel_size=kernel,
            ),
        )

        ds.to_netcdf(nc_path)
        print(f"  Wrote NetCDF: {nc_path}")

        # --- 7. Write CSV metadata file ---
        df = pd.DataFrame(
            {
                "id": tile_id,
                "x_start": x_start,
                "x_stop": x_stop,
                "y_start": y_start,
                "y_stop": y_stop,
                "x_coord": x_coord,
                "y_coord": y_coord,
                "fracture_density": fracture_density,
            }
        )
        csv_path = crev_dir / f"FracTiles_{image_id}.csv"
        df.to_csv(csv_path, index=False)
        print(f"  Wrote CSV: {csv_path}")

        total_tiles += n_tiles

    print(f"\nTotal tiles across all images: {total_tiles}")


# ============================
# Example usage
# ============================

#images = [
#    "WV01_20120802153817", "WV01_20120802153816", "WV01_20120802153815",
#    "WV01_20120802153814", "WV01_20120802153813", "WV01_20120713005153",
#    "WV01_20120713005152", "WV01_20120713005151", "WV01_20120803164856",
#    "QB02_20120729152314", "QB02_20120731154958", "QB02_20120731155001",
#    "QB02_20120731155004", "WV01_20120713164417", "WV01_20120713164418",
#    "WV01_20120713164419", "WV01_20120803164853", "WV01_20120803164854",
#    "WV01_20120803164855",
#]

images = [
    "tae_8_crevasse_gray",
]

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Where the full WorldView / optical image lives
img_dir = PROJECT_ROOT / "InputImages"          # tae_8_crevasse_gray.tif

# Where FracMap_*.tif and DataMask_*.tif live (MaskBorder outputs)
crev_dir = PROJECT_ROOT / "PostProcessing" / "Masked"

create_image_chips(images, img_dir=img_dir, crev_dir=crev_dir)
