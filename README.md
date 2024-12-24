# YOLO Dataset tiling 

## Tile (slice) YOLO Dataset for Small Objects Detection and Instance Segmentation

This module can cut images and corresponding labels from YOLO dataset into tiles of specified size and create a 
new dataset based on these tiles. It supports both object detection and instance segmentation. More details you can find 
in the <a href="https://supervision.roboflow.com/develop/detection/tools/inference_slicer/#supervision.detection.tools.inference_slicer.InferenceSlicer">docs</a>.

## Installation

To install the package, use pip:

```bash
pip install yolo-tiling
```

## Usage

```python
from yolo_tiler import YoloTiler, TileConfig

src = "path/to/dataset"  # Source YOLO dataset directory
dst = "path/to/tiled_dataset"  # Output directory for tiled dataset

config = TileConfig(
    # Size of each tile (width, height). Can be:
    # - Single integer for square tiles: slice_wh=640
    # - Tuple for rectangular tiles: slice_wh=(640, 480)
    slice_wh=(640, 480),

    # Overlap between adjacent tiles. Can be:
    # - Single float (0-1) for uniform overlap percentage: overlap_wh=0.1
    # - Tuple of floats for different overlap in each dimension: overlap_wh=(0.1, 0.1) 
    # - Single integer for pixel overlap: overlap_wh=64
    # - Tuple of integers for different pixel overlaps: overlap_wh=(64, 48)
    overlap_wh=(0.1, 0.1),

    # Image file extension to process
    ext=".png",

    # Type of YOLO annotations to process:
    # - "object_detection": Standard YOLO format (class, x, y, width, height)
    # - "instance_segmentation": YOLO segmentation format (class, x1, y1, x2, y2, ...)
    annotation_type="instance_segmentation",

    # For segmentation only: Controls point density along polygon edges
    # Lower values = more points, higher quality but larger files
    densify_factor=0.5,

    # For segmentation only: Controls polygon smoothing
    # Lower values = more details preserved, higher values = smoother shapes
    smoothing_tolerance=0.1,

    # Dataset split ratios (must sum to 1.0)
    train_ratio=0.7,  # Proportion of data for training
    valid_ratio=0.2,  # Proportion of data for validation
    test_ratio=0.1,   # Proportion of data for testing

    # Optional margins to exclude from input images. Can be:
    # - Single float (0-1) for uniform margin percentage: margins=0.1
    # - Tuple of floats for different margins: margins=(0.1, 0.1, 0.1, 0.1)
    # - Single integer for pixel margins: margins=64
    # - Tuple of integers for different pixel margins: margins=(64, 64, 64, 64)
    margins=0.0
)

tiler = YoloTiler(
    source=src,
    target=dst,
    config=config,
)

tiler.run()
```

The tiler requires a YOLO dataset structure in both source and target directories. If only a `train` folder exists, the train / valid / test ratios will be used to split the tiled `train` folder; else, the ratios are ignored.

```bash
dataset/
├── train/
│   ├── images/
│   └── labels/
├── valid/
│   ├── images/
│   └── labels/
├── test/
│   ├── images/
│   └── labels/
└── data.yaml  # Optional
```

## Command Line Usage

You can also use the command line interface to run the tiling process. Here are the instructions:

```bash
python src/yolo_tiler.py <source> <target> [--slice_wh SLICE_WH SLICE_WH] [--overlap_wh OVERLAP_WH OVERLAP_WH] [--ext EXT] [--annotation_type ANNOTATION_TYPE] [--densify_factor DENSIFY_FACTOR] [--smoothing_tolerance SMOOTHING_TOLERANCE] [--train_ratio TRAIN_RATIO] [--valid_ratio VALID_RATIO] [--test_ratio TEST_RATIO]
```

### Example Commands

1. Basic usage with default parameters:
```bash
python src/yolo_tiler.py path/to/dataset path/to/tiled_dataset
```

2. Custom slice size and overlap:
```bash
python src/yolo_tiler.py path/to/dataset path/to/tiled_dataset --slice_wh 640 480 --overlap_wh 0.1 0.1
```

3. Custom annotation type and image extension:
```bash
python src/yolo_tiler.py path/to/dataset path/to/tiled_dataset --annotation_type instance_segmentation --ext .jpg
```

### Memory Efficiency

The `tile_image` method now uses rasterio's Window to read and process image tiles directly from the disk, instead of loading the entire image into memory. This makes the tiling process more memory efficient, especially for large images.