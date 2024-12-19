# YOLO Dataset tiling script

## Tile (slice) YOLO Dataset for Small Objects Detection and Instance Segmentation

This script can cut images and corresponding labels from YOLO dataset into tiles of specified size and create a new dataset based on these tiles. It supports both object detection and instance segmentation. More details you can find in the <a href="https://towardsdatascience.com/tile-slice-yolo-dataset-for-small-objects-detection-a75bf26f7fa2">article</a>.

## Installation

To install the package, use pip:

```bash
pip install yolo-tiling
```

## Usage 

`python3 -m yolo_tiling -source ./yolosample/ts/ -target ./yolosliced/ts/ -ext .JPG -size 512`

## Arguments

- **-source**        Source folder with images and labels needed to be tiled. Default: ./yolosample/ts/
- **-target**        Target folder for a new sliced dataset. Default: ./yolosliced/ts/
- **-ext**           Image extension in a dataset. Default: .JPG
- **-falsefolder**   Folder for tiles without bounding boxes
- **-size**          Size of a tile. Default: 416
- **-ratio**         Train/test split ratio. Dafault: 0.8

## Instance Segmentation Usage

To tile instance segmentation datasets, use the following command:

`python3 -m yolo_tiling -source ./yolosample/ts/ -target ./yolosliced/ts/ -ext .JPG -size 512 -annotation_type instance_segmentation`

## Class-based Usage

To use the script as a class, you can create an instance of the `YoloTiler` class and call its `run` method:

```python
from yolo_tiling import YoloTiler

yolo_tiler = YoloTiler(source="./yolosample/ts/", target="./yolosliced/ts/", ext=".JPG", falsefolder=None, size=512, ratio=0.8)
yolo_tiler.run()
```

## Class-based Usage for Instance Segmentation

To use the script as a class for instance segmentation, you can create an instance of the `YoloTiler` class and call its `run` method with the `annotation_type` argument set to `instance_segmentation`:

```python
from yolo_tiling import YoloTiler

yolo_tiler = YoloTiler(source="./yolosample/ts/", target="./yolosliced/ts/", ext=".JPG", falsefolder=None, size=512, ratio=0.8, annotation_type="instance_segmentation")
yolo_tiler.run()
```
