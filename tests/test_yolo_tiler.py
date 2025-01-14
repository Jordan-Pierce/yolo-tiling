#!/usr/bin/env python

"""Tests for `yolo_tiler` package."""

from yolo_tiler import YoloTiler, TileConfig, TileProgress


def progress_callback(progress: TileProgress):
    # Determine whether to show tile or image progress
    if progress.total_tiles > 0:
        print(f"Processing {progress.current_image_name} in {progress.current_set_name} set: "
              f"Tile {progress.current_tile_idx}/{progress.total_tiles}")
    else:
        print(f"Processing {progress.current_image_name} in {progress.current_set_name} set: "
              f"Image {progress.current_image_idx}/{progress.total_images}")
        
src = "./tests/detection_tiled"
dst = "./tests/detection_tiled_tiled"

config = TileConfig(
    slice_wh=(320, 240),  # Slice width and height
    overlap_wh=(0.0, 0.0),  # Overlap width and height (10% overlap in this example, or 64x48 pixels)
    input_ext=".png",
    output_ext=None,
    annotation_type="object_detection",
    train_ratio=0.7,
    valid_ratio=0.2,
    test_ratio=0.1,
    margins=(0, 0, 0, 0),  # Left, top, right, bottom
    include_negative_samples=True  # Inlude negative samples
)


# Create tiler with callback
tiler = YoloTiler(
    source=src,
    target=dst,
    config=config,
    num_viz_samples=25,
    progress_callback=progress_callback
)

# Run tiling process
tiler.run()
