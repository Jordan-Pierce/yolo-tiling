import os
import unittest
from unittest.mock import patch, MagicMock
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

class TestYoloTiler(unittest.TestCase):

    @patch('yolo_tiler.YoloTiler._save_tile')
    def test_logging_functionality(self, mock_save_tile):
        with self.assertLogs('YoloTiler', level='INFO') as log:
            tiler.run()
            self.assertIn('INFO:YoloTiler:Saved tile image to', log.output[0])
            self.assertIn('INFO:YoloTiler:Saved tile labels to', log.output[1])

    @patch('yolo_tiler.tqdm')
    def test_progress_bar(self, mock_tqdm):
        tiler.run()
        self.assertTrue(mock_tqdm.called)

if __name__ == '__main__':
    unittest.main()

