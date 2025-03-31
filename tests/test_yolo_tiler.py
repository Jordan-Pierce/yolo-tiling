#!/usr/bin/env python

"""Tests for `yolo_tiler` package."""

from yolo_tiler import YoloTiler
from yolo_tiler import TileConfig
from yolo_tiler import TileProgress

# ----------------------------------------------------------------------------------------------------------------------
# Functions
# ----------------------------------------------------------------------------------------------------------------------


def progress_callback(progress: TileProgress):
    # Determine whether to show tile or image progress
    if progress.total_tiles > 0:
        print(f"Processing {progress.current_image_name} in {progress.current_set_name} set: "
              f"Tile {progress.current_tile_idx}/{progress.total_tiles}")
    else:
        print(f"Processing {progress.current_image_name} in {progress.current_set_name} set: "
              f"Image {progress.current_image_idx}/{progress.total_images}")
        

# ----------------------------------------------------------------------------------------------------------------------
# Tests
# ----------------------------------------------------------------------------------------------------------------------
test_detection = True
test_classification = True
test_segmentation = True

if test_detection:
    
    src_detection = "./tests/detection"
    dst_detection = "./tests/detection_tiled"

    config_detection = TileConfig(
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

    # Create tiler with callback for object detection
    tiler_detection = YoloTiler(
        source=src_detection,
        target=dst_detection,
        config=config_detection,
        num_viz_samples=5,
        progress_callback=progress_callback
    )

    # Run tiling process for object detection
    tiler_detection.run()

if test_classification:
    
    src_classification = "./tests/classification"
    dst_classification = "./tests/classification_tiled"

    config_classification = TileConfig(
        slice_wh=(320, 240),  # Slice width and height
        overlap_wh=(0.0, 0.0),  # Overlap width and height (10% overlap in this example, or 64x48 pixels)
        input_ext=".jpg",
        output_ext=None,
        annotation_type="image_classification",
        train_ratio=0.7,
        valid_ratio=0.2,
        test_ratio=0.1,
        margins=(0, 0, 0, 0),  # Left, top, right, bottom
        include_negative_samples=True  # Inlude negative samples
    )

    # Create tiler with callback for image classification
    tiler_classification = YoloTiler(
        source=src_classification,
        target=dst_classification,
        config=config_classification,
        num_viz_samples=5,
        progress_callback=progress_callback
    )

    # Run tiling process for image classification
    tiler_classification.run()
    
if test_segmentation:

    src_segmentation = "./tests/segmentation"
    dst_segmentation = "./tests/segmentation_tiled"

    config_segmentation = TileConfig(
        slice_wh=(320, 240),  # Slice width and height
        overlap_wh=(0.0, 0.0),  # Overlap width and height (10% overlap in this example, or 64x48 pixels)
        input_ext=".png",
        output_ext=None,
        annotation_type="instance_segmentation",
        train_ratio=0.7,
        valid_ratio=0.2,
        test_ratio=0.1,
        margins=(0, 0, 0, 0),  # Left, top, right, bottom
        include_negative_samples=True  # Inlude negative samples
    )

    # Create tiler with callback for instance segmentation
    tiler_segmentation = YoloTiler(
        source=src_segmentation,
        target=dst_segmentation,
        config=config_segmentation,
        num_viz_samples=5,
        progress_callback=progress_callback
    )

    # Run tiling process for instance segmentation
    tiler_segmentation.run()
