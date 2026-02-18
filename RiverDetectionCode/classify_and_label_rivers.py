with rasterio.open(image_file) as src:
    img = src.read(1)
    transform = src.transform
    crs = src.crs
    profile = src.profile.copy()
    src_tags = src.tags()
    src_band1_tags = src.tags(1)

binary = img.astype(np.float32) > float(threshold)
labeled = label(binary, connectivity=2).astype(np.int32)

output_file = outputpath / f"{name}_river{ext}"

profile.update(
    dtype="int32",
    count=1,
    height=labeled.shape[0],
    width=labeled.shape[1],
    transform=transform,
    crs=crs,
)

with rasterio.open(output_file, "w", **profile) as dst:
    dst.write(labeled, 1)
    ...
