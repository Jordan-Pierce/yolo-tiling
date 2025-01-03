#!/usr/bin/env python

"""Tests for `yolo_tiler` package."""

from yolo_tiler import YoloTiler, TileConfig, TileProgress


def progress_callback(progress: TileProgress):
    print(f"Processing {progress.current_image} in {progress.current_set} set: "
          f"tile {progress.current_tile}/{progress.total_tiles}")


src = "./tests/segmentation"
dst = "./tests/segmentation_tiled"

config = TileConfig(
    slice_wh=(640, 480),  # Slice width and height
    overlap_wh=(0.1, 0.1),  # Overlap width and height (10% overlap in this example, or 64x48 pixels)
    ext=".png",
    annotation_type="instance_segmentation",
    train_ratio=0.7,
    valid_ratio=0.2,
    test_ratio=0.1,
    margins=(10, 10, 10, 10),  # Left, top, right, bottom
)


# Create tiler with callback
tiler = YoloTiler(
    source=src,
    target=dst,
    config=config,
    callback=progress_callback
)

# Run tiling process
tiler.run()