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

src = "path/to/dataset"
dst = "path/to/tiled_dataset"

config = TileConfig(
    slice_wh=(640, 480),  # Slice width and height
    overlap_wh=(64, 48),  # Overlap width and height (10% overlap in this example)
    ext=".png",
    annotation_type="instance_segmentation",
    train_ratio=0.7,
    valid_ratio=0.2,
    test_ratio=0.1
)

tiler = YoloTiler(
    source=src,
    target=dst,
    config=config,
)

tiler.run()
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
python src/yolo_tiler.py path/to/dataset path/to/tiled_dataset --slice_wh 640 480 --overlap_wh 64 48
```

3. Custom annotation type and image extension:
```bash
python src/yolo_tiler.py path/to/dataset path/to/tiled_dataset --annotation_type instance_segmentation --ext .jpg
```

## Note
g
The source and target folders must be YOLO formatted with `train`, `val`, `test` subfolders, each containing 
`images/` and `labels/` subfolders.