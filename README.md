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

## Note

The source and target folders must be YOLO formatted with `train`, `val`, `test` subfolders, each containing 
`images/` and `labels/` subfolders.

## New Features

### Progress Bar

The `run` method now includes a progress bar for better user experience. This progress bar provides a visual indication of the tiling process, making it easier to track the progress.

### Enhanced Logging

The logging functionality has been improved to provide more detailed information and error handling. This includes logging the saved tile path and labels, as well as any errors that occur during the tiling process.