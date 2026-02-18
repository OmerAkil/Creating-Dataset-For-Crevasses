from pathlib import Path
import rasterio
from rasterio.crs import CRS


def make_compatible_images(
    target_epsg=3413,
):
    """
    Python equivalent of MakeCompatibleImages.m

    Reads all .tif files from:
        CrevasseCNN/Images/
    Writes CRS-normalized GeoTIFFs to:
        CrevasseCNN/InputImages/
    """

    # Resolve paths based on PROJECT ROOT
    PROJECT_ROOT = Path(__file__).resolve().parent.parent
    in_dir = PROJECT_ROOT / "Images"
    out_dir = PROJECT_ROOT / "InputImages"

    out_dir.mkdir(parents=True, exist_ok=True)

    crs = CRS.from_epsg(target_epsg)

    tif_files = sorted(in_dir.glob("*.tif"))
    print(f"Found {len(tif_files)} .tif files in {in_dir}")

    for tif_path in tif_files:
        print(tif_path.name)
        with rasterio.open(tif_path) as src:
            data = src.read()
            profile = src.profile.copy()

        # Enforce GeoTIFF + EPSG:3413 CRS
        profile.update(
            driver="GTiff",
            crs=crs
        )

        #base_out_name = tif_path.name[:19] + ".tif"
        base_out_name = tif_path.name[:19]
        out_file = out_dir / base_out_name

        with rasterio.open(out_file, "w", **profile) as dst:
            dst.write(data)

    print("Done. Compatible images written to:", out_dir)


if __name__ == "__main__":
    make_compatible_images()
