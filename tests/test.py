from yolo_tiler import YoloTiler, TileConfig

src = "C:/Users/jordan.pierce/Downloads/TagLab/sampleProjects/data/segmentation"
dst = "C:/Users/jordan.pierce/Downloads/TagLab/sampleProjects/data/segmentation_tiled"

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

tiler = YoloTiler(
    source=src,
    target=dst,
    config=config,
)

tiler.run()