import pandas as pd
import numpy as np
from PIL import Image
from shapely.geometry import Polygon, MultiPolygon
from pathlib import Path
from typing import List, Tuple, Optional, Union, Generator
from dataclasses import dataclass
import glob
from shutil import copyfile
import logging
import math
import yaml


@dataclass
class TileConfig:
    """Configuration for tiling parameters"""
    # (width, height) of each slice
    slice_wh: Tuple[int, int]  
    # (width, height) overlap between slices
    overlap_wh: Tuple[int, int]  
    # train/test split ratio
    ratio: float  
    # image extension
    ext: str  
    # type of annotation
    annotation_type: str = "object_detection"  

    def __post_init__(self):
        """Validate configuration parameters"""
        if not all(x > 0 for x in self.slice_wh):
            raise ValueError("Slice dimensions must be positive")
        if not all(x >= 0 for x in self.overlap_wh):
            raise ValueError("Overlap must be non-negative")
        if not all(o < s for o, s in zip(self.overlap_wh, self.slice_wh)):
            raise ValueError("Overlap must be less than slice size")


class YoloTiler:
    """
    A class to tile YOLO dataset images and their corresponding annotations.
    Supports both object detection and instance segmentation formats.
    """

    def __init__(self, 
                 source: Union[str, Path], 
                 target: Union[str, Path],
                 config: TileConfig):
        """
        Initialize YoloTiler with source and target directories.

        Args:
            source: Source directory containing YOLO dataset
            target: Target directory for sliced dataset
            config: TileConfig object containing tiling parameters
            false_folder: Optional directory for storing tiles without annotations
        """
        self.source = Path(source)
        self.target = Path(target)
        self.false_folder = Path(self.target / "false_folder")
        self.config = config
        self.logger = self._setup_logger()
        
        self.subfolders = ['train/', 
                           'valid/', 
                           'test/']

    def _setup_logger(self) -> logging.Logger:
        """Configure logging for the tiler"""
        logger = logging.getLogger('YoloTiler')
        logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        return logger
    
    def _create_target_folder(self, target: Path) -> None:
        """Create target folder if it does not exist"""
        for subfolder in self.subfolders:
            (target / subfolder / "images").mkdir(parents=True, exist_ok=True)
            (target / subfolder / "labels").mkdir(parents=True, exist_ok=True)
            
        # Create false folder if needed
        self.false_folder.mkdir(parents=True, exist_ok=True)
        
    def _validate_yolo_structure(self, folder: Path) -> None:
        """
        Validate YOLO dataset folder structure.

        Args:
            folder: Path to check for YOLO structure

        Raises:
            ValueError: If required folders are missing
        """
        for subfolder in self.subfolders:
            if not (folder / subfolder / 'images').exists():
                raise ValueError(f"Required folder {folder / subfolder / 'images'} does not exist")
            if not (folder / subfolder / 'labels').exists():
                raise ValueError(f"Required folder {folder / subfolder / 'labels'} does not exist")

    def _calculate_tile_positions(self, 
                                  image_size: Tuple[int, int]) -> Generator[Tuple[int, int, int, int], None, None]:
        """
        Calculate tile positions with overlap.
        
        Args:
            image_size: (width, height) of the image
            
        Yields:
            Tuples of (x1, y1, x2, y2) for each tile
        """
        img_w, img_h = image_size
        slice_w, slice_h = self.config.slice_wh
        overlap_w, overlap_h = self.config.overlap_wh

        # Calculate effective step sizes
        step_w = slice_w - overlap_w
        step_h = slice_h - overlap_h

        # Calculate number of tiles in each dimension
        num_tiles_w = math.ceil((img_w - overlap_w) / step_w)
        num_tiles_h = math.ceil((img_h - overlap_h) / step_h)

        for i in range(num_tiles_h):
            for j in range(num_tiles_w):
                # Calculate tile coordinates
                x1 = j * step_w
                y1 = i * step_h
                x2 = min(x1 + slice_w, img_w)
                y2 = min(y1 + slice_h, img_h)

                # Handle edge cases by shifting tiles
                if x2 == img_w and x2 != x1 + slice_w:
                    x1 = max(0, x2 - slice_w)
                if y2 == img_h and y2 != y1 + slice_h:
                    y1 = max(0, y2 - slice_h)

                yield (x1, y1, x2, y2)
                
    def _normalize_coordinates(self, 
                               coords: List[Tuple[float, float]], 
                               tile_bounds: Tuple[int, int, int, int]) -> List[str]:
            """
            Normalize coordinates to [0,1] range relative to tile bounds.
            
            Args:
                coords: List of (x,y) coordinates
                tile_bounds: (x1, y1, x2, y2) of tile bounds
                
            Returns:
                List of normalized coordinate strings
            """
            x1, y1, x2, y2 = tile_bounds
            tile_width = x2 - x1
            tile_height = y2 - y1
            
            normalized = []
            for x, y in coords:
                normalized.append(f"{(x - x1) / tile_width:.6f},{(y - y1) / tile_height:.6f}")
                
            return normalized

    def _process_intersection(self, intersection: Union[Polygon, MultiPolygon]) -> List[Tuple[float, float]]:
        """
        Process intersection geometry to get coordinate list.
        
        Args:
            intersection: Shapely geometry object
            
        Returns:
            List of coordinates
        """
        if isinstance(intersection, Polygon):
            return list(intersection.exterior.coords)[:-1]
        return list(max(intersection.geoms, key=lambda p: p.area).exterior.coords)[:-1]

    def _save_labels(self, labels: List, path: Path, is_segmentation: bool) -> None:
        """
        Save labels to file in appropriate format.

        Args:
            labels: List of label data
            path: Path to save labels
            is_segmentation: Whether using segmentation format
        """
        if is_segmentation:
            with open(path, 'w') as f:
                for label_class, points in labels:
                    f.write(f"{label_class} {points}\n")
        else:
            df = pd.DataFrame(labels, columns=['class', 'x1', 'y1', 'w', 'h'])
            df.to_csv(path, sep=' ', index=False, header=False, float_format='%.6f')

    def tile_image(self, image_path: Path, label_path: Path, folder: str) -> None:
        """
        Tile an image and its corresponding labels.

        Args:
            image_path: Path to image file
            label_path: Path to label file
            folder: Subfolder name (train, valid, test) for image and label files
        """
        # Read image and labels
        image = Image.open(image_path)
        image_array = np.array(image, dtype=np.uint8)
        width, height = image.size

        # Read labels based on annotation type
        if self.config.annotation_type == "object_detection":
            columns = ['class', 'x1', 'y1', 'w', 'h']
        else:
            columns = ['class', 'points']

        # Read the label file
        labels = pd.read_csv(label_path, sep=' ', names=columns)

        # Process annotations
        boxes = []
        for _, row in labels.iterrows():
            if self.config.annotation_type == "object_detection":
                # Convert YOLO format to pixel coordinates
                x_center, y_center = row['x1'] * width, row['y1'] * height
                box_w, box_h = row['w'] * width, row['h'] * height
                x1 = x_center - box_w / 2
                y1 = y_center - box_h / 2
                x2 = x_center + box_w / 2
                y2 = y_center + box_h / 2
                boxes.append((int(row['class']), Polygon([(x1, y1), (x2, y1), (x2, y2), (x1, y2)])))
            else:
                # Process instance segmentation points
                points = []
                for point in row['points'].split(';'):
                    for x, y in [point.split(',')]:
                        points.append((float(x) * width, float(y) * height))
                        
                boxes.append((int(row['class']), Polygon(points)))

        # Process each tile
        for tile_idx, (x1, y1, x2, y2) in enumerate(self._calculate_tile_positions((width, height))):
            tile_data = image_array[y1:y2, x1:x2]
            tile_polygon = Polygon([(x1, y1), (x2, y1), (x2, y2), (x1, y2)])
            tile_labels = []
            has_annotations = False

            # Process annotations for this tile
            for box_class, box_polygon in boxes:
                if tile_polygon.intersects(box_polygon):
                    has_annotations = True
                    intersection = tile_polygon.intersection(box_polygon)
                    
                    if self.config.annotation_type == "instance_segmentation":
                        # Handle instance segmentation
                        coords = self._process_intersection(intersection)
                        normalized = self._normalize_coordinates(coords, (x1, y1, x2, y2))
                        tile_labels.append([box_class, ";".join(normalized)])
                    else:
                        # Handle object detection
                        bbox = intersection.envelope
                        center = bbox.centroid
                        bbox_coords = bbox.exterior.coords.xy
                        new_width = (max(bbox_coords[0]) - min(bbox_coords[0])) / (x2 - x1)
                        new_height = (max(bbox_coords[1]) - min(bbox_coords[1])) / (y2 - y1)
                        new_x = (center.x - x1) / (x2 - x1)
                        new_y = (center.y - y1) / (y2 - y1)
                        tile_labels.append([box_class, new_x, new_y, new_width, new_height])

            # Save tile and annotations
            tile_suffix = f'_tile_{tile_idx}{self.config.ext}'
            if has_annotations:
                self._save_tile(tile_data, image_path, tile_suffix, tile_labels, folder, False)
            elif self.false_folder:
                self._save_tile(tile_data, image_path, tile_suffix, None, folder, True)
                
    def _process_tile(self, 
                      image: np.ndarray, 
                      boxes: List[Tuple[int, Polygon]],
                      i: int, 
                      j: int, 
                      image_path: Path, 
                      height: int) -> None:
        """
        Process a single tile from the image.

        Args:
            image: Full image array
            boxes: List of (class, polygon) tuples
            i, j: Tile indices
            image_path: Path to original image
            height: Image height
        """
        tile_size = self.config.size
        x1 = j * tile_size
        y1 = height - (i * tile_size)
        x2 = ((j + 1) * tile_size) - 1
        y2 = (height - (i + 1) * tile_size) + 1

        tile_polygon = Polygon([(x1, y1), (x2, y1), (x2, y2), (x1, y2)])
        tile_labels = []
        tile_saved = False

        for box_class, box_polygon in boxes:
            if tile_polygon.intersects(box_polygon):
                if not tile_saved:
                    tile_image = image[i * tile_size: (i + 1) * tile_size, j * tile_size: (j + 1) * tile_size]
                    self._save_tile_image(tile_image, image_path, i, j)
                    tile_saved = True

                intersection = tile_polygon.intersection(box_polygon)
                if self.config.annotation_type == "instance_segmentation":
                    coords = self._process_intersection(intersection)
                    normalized = self._normalize_coordinates(coords, (x1, y1, x2, y2), tile_size)
                    tile_labels.append([box_class, ";".join(normalized)])
                else:
                    bbox = intersection.envelope
                    center = bbox.centroid
                    x, y = bbox.exterior.coords.xy
                    new_width = (max(x) - min(x)) / tile_size
                    new_height = (max(y) - min(y)) / tile_size
                    new_x = (center.coords.xy[0][0] - x1) / tile_size
                    new_y = (y1 - center.coords.xy[1][0]) / tile_size
                    tile_labels.append([box_class, new_x, new_y, new_width, new_height])

        if tile_labels:
            output_path = self.target / image_path.with_suffix('.txt').name.replace(self.config.ext, f'_{i}_{j}.txt')
            self._save_labels(tile_labels, output_path, self.config.annotation_type == "instance_segmentation")
            
        elif not tile_saved and self.false_folder:
            tile_image = image[i * tile_size: (i + 1) * tile_size, j * tile_size: (j + 1) * tile_size]
            self._save_tile_image(tile_image, image_path, i, j, is_false=True)

    def _save_tile(self,
                   tile_data: np.ndarray, 
                   original_path: Path, 
                   suffix: str, 
                   labels: Optional[List],
                   folder: str,
                   is_false: bool = False) -> None:
        """
        Save a tile image and its labels.
        
        Args:
            tile_data: Numpy array of tile image
            original_path: Path to original image
            suffix: Suffix for the tile filename
            labels: List of labels for the tile
            folder: Subfolder name (train, valid, test) for image and label files
            is_false: Whether to save to false folder
        """
        # Set the save directory
        save_dir = self.false_folder if is_false else self.target / folder

        # Save the image in the appropriate directory
        image_path = save_dir / "images" / original_path.name.replace(self.config.ext, suffix)
        Image.fromarray(tile_data).save(image_path)
        
        if labels:
            # Save the labels in the appropriate directory
            label_path = save_dir / "labels" / original_path.name.replace(self.config.ext, suffix).with_suffix('.txt')
            is_segmentation = self.config.annotation_type == "instance_segmentation"
            if is_segmentation:
                with open(label_path, 'w') as f:
                    for label_class, points in labels:
                        f.write(f"{label_class} {points}\n")
            else:
                df = pd.DataFrame(labels, columns=['class', 'x1', 'y1', 'w', 'h'])
                df.to_csv(label_path, sep=' ', index=False, header=False, float_format='%.6f')

    def _save_tile_image(self, 
                         tile_array: np.ndarray, 
                         original_path: Path, 
                         i: int, 
                         j: int, 
                         is_false: bool = False) -> None:
        """
        Save a tile image to the appropriate directory.

        Args:
            tile_array: Numpy array of tile image
            original_path: Path to original image
            i, j: Tile indices
            is_false: Whether to save to false folder
        """
        if is_false and not self.false_folder:
            return
        
        if is_false:
            save_path = self.false_folder / original_path.name.replace(self.config.ext, f'_{i}_{j}{self.config.ext}')
        else:
            save_path = self.target / original_path.name.replace(self.config.ext, f'_{i}_{j}{self.config.ext}')
        
        Image.fromarray(tile_array).save(save_path)
        self.logger.info(f"Saved tile to {save_path}")

    def split_dataset(self) -> None:
        """Split the dataset into train and test sets"""
        image_paths = list(self.target.glob(f'*{self.config.ext}'))
        np.random.shuffle(image_paths)
        split_idx = int(len(image_paths) * self.config.ratio)

        train_paths = image_paths[:split_idx]
        test_paths = image_paths[split_idx:]

        self.logger.info(f'Train set: {len(train_paths)} images')
        self.logger.info(f'Test set: {len(test_paths)} images')

        for filename, paths in [('train.txt', train_paths), ('test.txt', test_paths)]:
            with open(self.target.parent / filename, 'w') as f:
                f.write('\n'.join(str(p) for p in paths))

    def run(self) -> None:
        """Run the complete tiling process"""
        try:
            # Validate directories
            self._validate_yolo_structure(self.source)
            self._create_target_folder(self.target)

            # Train, valid, test subfolders
            for subfolder in self.subfolders:
                image_paths = list((self.source / subfolder / 'images').glob(f'*{self.config.ext}'))
                label_paths = list((self.source / subfolder / 'labels').glob('*.txt'))
                    
                # Log the number of images, labels found
                self.logger.info(f'Found {len(image_paths)} images in {subfolder} directory')
                self.logger.info(f'Found {len(label_paths)} label files in {subfolder} directory')

                # Check for missing files
                if not image_paths:
                    raise ValueError("No images found in source directory")
                if len(image_paths) != len(label_paths):
                    raise ValueError("Unequal number of images and label files")

                # Process each image
                for image_path, label_path in list(zip(image_paths, label_paths)):
                    assert image_path.stem == label_path.stem, "Image and label filenames do not match"
                    self.logger.info(f'Processing {image_path}')
                    self.tile_image(image_path, label_path, subfolder)

            # Split dataset
            self.split_dataset()
            self.logger.info('Tiling process completed successfully')
            
            # Copy classes from data.yaml if it exists
            data_yaml = self.source / 'data.yaml'
            if data_yaml.exists():
                with open(data_yaml, 'r') as f:
                    data = yaml.safe_load(f)
                if 'names' in data:
                    with open(self.target.parent / 'classes.names', 'w') as f:
                        f.write('\n'.join(data['names']))
                else:
                    self.logger.warning('No classes found in data.yaml')
            else:
                self.logger.warning('data.yaml not found in source directory')

        except Exception as e:
            self.logger.error(f'Error during tiling process: {str(e)}')
            raise