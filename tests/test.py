import os
from yolo_tiler import YoloTiler, TileConfig

src = "C:/Users/jordan.pierce/Downloads/TagLab/sampleProjects/data/segmentation"
dst = "C:/Users/jordan.pierce/Downloads/TagLab/sampleProjects/data/segmentation_tiled"

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