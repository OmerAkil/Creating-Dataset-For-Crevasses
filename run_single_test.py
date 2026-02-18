from pathlib import Path
import sys

# --- Project root (where this file lives) ---
PROJECT_ROOT = Path(__file__).resolve().parent

# --- Make RiverDetectionCode importable as plain modules ---
RIVER_CODE_DIR = PROJECT_ROOT / "RiverDetectionCode"
if str(RIVER_CODE_DIR) not in sys.path:
    sys.path.append(str(RIVER_CODE_DIR))

# Now we can import run_river_detection, which itself imports
# spectralanalysis, multidirection_gabor, im_pathopening, etc.
from run_river_detection import run_river_detection

# CHANGE THIS to your actual filename in InputImages
IMAGE_NAME = "tae_8.tif"  # e.g. "WV01_20120803.tif"

image_file = PROJECT_ROOT / "InputImages" / IMAGE_NAME
output_dir = PROJECT_ROOT / "RiverOutput"

# Choose the correct sensor for your image:
# "WV", "SPOT", "SETSM", "Sentinel2", "Landsat", "LandsatNDWI"
SENSOR = "WV"        # change if needed
INVERSE = 1          # 1 = dark rivers -> bright; 0 = already bright (e.g. NDWI)

run_river_detection(
    image_file=image_file,
    outputpath=output_dir,
    sensor=SENSOR,
    inverse=INVERSE,
    width=2,
    ppolength=20,
    histCountThreshold=100,
)

