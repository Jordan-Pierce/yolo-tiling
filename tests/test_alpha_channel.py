import numpy as np
import rasterio
from PIL import Image

from yolo_tiler import TileConfig, YoloTiler


def test_rgba_tiles_are_saved_without_alpha_channel(tmp_path):
    source_dir = tmp_path / "source"
    target_dir = tmp_path / "target"
    image_dir = source_dir / "train" / "images"
    label_dir = source_dir / "train" / "labels"
    image_dir.mkdir(parents=True)
    label_dir.mkdir(parents=True)

    image_path = image_dir / "rgba.png"
    rgba = np.zeros((8, 8, 4), dtype=np.uint8)
    rgba[..., 0] = 40
    rgba[..., 1] = 80
    rgba[..., 2] = 120
    rgba[..., 3] = 200
    Image.fromarray(rgba, mode="RGBA").save(image_path)

    (label_dir / "rgba.txt").write_text("0 0.5 0.5 1 1\n", encoding="utf-8")

    tiler = YoloTiler(
        source=source_dir,
        target=target_dir,
        config=TileConfig(
            slice_wh=(8, 8),
            overlap_wh=(0, 0),
            annotation_type="object_detection",
            output_ext=".png",
            include_negative_samples=True,
        ),
        num_viz_samples=0,
        show_processing_status=False,
    )

    try:
        tiler._process_subfolder("train/")

        output_files = list((target_dir / "train" / "images").glob("*.png"))
        assert len(output_files) == 1

        with rasterio.open(output_files[0]) as src:
            assert src.count == 3
    finally:
        tiler.save_executor.shutdown(wait=True)