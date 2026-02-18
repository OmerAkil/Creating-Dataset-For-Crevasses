import numpy as np
from mat2gray import mat2gray


def histCountCut(image, threshold):
    """
    Python translation of histCountCut.m

    Parameters
    ----------
    image : ndarray (ideally uint8)
        Input image. Values are assumed to be in [0, 255].
    threshold : int
        Pixel-count threshold for cutting histogram tails.

    Returns
    -------
    output_image : ndarray, uint8
        Image with histogram tails clipped and rescaled to [0, 255].
    """
    img = np.asarray(image).copy()

    # Flatten for histogram computation
    # MATLAB histcounts(image,256) -> 256 bins over the data range;
    # here we assume intensity range [0, 255] for uint8 images.
    hist, bin_edges = np.histogram(img, bins=256, range=(0, 256))

    # Initialize bounds
    minBar = 0
    maxBar = 255

    # --- Find minBar ---
    # MATLAB:
    # for i=1:256
    #   if hist(i)<threshold, continue; end
    #   minBar = i - 1; break; end
    for i in range(256):
        if hist[i] < threshold:
            continue
        minBar = i  # i-1 in MATLAB, but with range(256) we already start at 0
        break

    # --- Find maxBar ---
    # MATLAB:
    # for i=1:256
    #   j = 256 - i;
    #   if hist(j)<threshold, continue; end
    #   maxBar = j + 1; break; end
    for i in range(256):
        j = 255 - i  # j goes 255, 254, ..., 0
        if hist[j] < threshold:
            continue
        maxBar = j   # j+1 in MATLAB with 1-based indexing; here we use j directly
        break

    # Clamp pixel values to [minBar, maxBar]
    img = np.clip(img, minBar, maxBar)

    # MATLAB:
    # output_image = image;
    # output_image=mat2gray(output_image);
    # output_image=uint8(output_image*255);
    output_image = mat2gray(img)
    output_image = (output_image * 255.0).astype(np.uint8)

    return output_image
