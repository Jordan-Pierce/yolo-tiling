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
from yolo_tiling import YoloTiler

config = TileConfig(
    slice_wh=(640, 480),  # Slice width and height
    overlap_wh=(64, 48),  # Overlap width and height (10% overlap in this example)
    ratio=0.8,  # Train/test split ratio
    ext=".jpg",
    annotation_type="object_detection"
)

tiler = YoloTiler(
    source="./yolosample/ts/",
    target="./yolosliced/ts/",
    config=config,
    false_folder="./false_tiles/"
)

tiler.run()
```

## Note

The source and target folders must be YOLO formatted with `train`, `val`, `test` subfolders, each containing 
`images/` and `labels/` subfolders.
