import os
from yolo_tiler import YoloTiler, TileConfig

src = "C:/Users/jordan.pierce/Downloads/TagLab/sampleProjects/data/segmentation"
dst = "C:/Users/jordan.pierce/Downloads/TagLab/sampleProjects/data/segmentation_tiled"

config = TileConfig(
    slice_wh=(640, 480),  # Slice width and height
    overlap_wh=(64, 48),  # Overlap width and height (10% overlap in this example)
    ratio=0.8,  # Train/test split ratio
    ext=".png",
    annotation_type="instance_segmentation",
)

tiler = YoloTiler(
    source=src,
    target=dst,
    config=config,
)

tiler.run()