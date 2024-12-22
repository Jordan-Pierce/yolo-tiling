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

def test_split_data():
    src = "C:/Users/jordan.pierce/Downloads/TagLab/sampleProjects/data/segmentation"
    dst = "C:/Users/jordan.pierce/Downloads/TagLab/sampleProjects/data/segmentation_tiled_split"

    config = TileConfig(
        slice_wh=(640, 480),  # Slice width and height
        overlap_wh=(64, 48),  # Overlap width and height (10% overlap in this example)
        ratio=0.8,  # Train/test split ratio
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

    tiler.split_data()

    # Check if the split data exists in the target directory
    assert os.path.exists(os.path.join(dst, 'train', 'images'))
    assert os.path.exists(os.path.join(dst, 'valid', 'images'))
    assert os.path.exists(os.path.join(dst, 'test', 'images'))

    train_images = os.listdir(os.path.join(dst, 'train', 'images'))
    valid_images = os.listdir(os.path.join(dst, 'valid', 'images'))
    test_images = os.listdir(os.path.join(dst, 'test', 'images'))

    # Check if the split ratios are approximately correct
    total_images = len(train_images) + len(valid_images) + len(test_images)
    assert abs(len(train_images) / total_images - 0.7) < 0.1
    assert abs(len(valid_images) / total_images - 0.2) < 0.1
    assert abs(len(test_images) / total_images - 0.1) < 0.1

test_split_data()
