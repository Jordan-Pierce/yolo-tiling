import warnings
import argparse
import logging
import math
import random
from dataclasses import dataclass
from pathlib import Path
from shutil import copyfile
from typing import List, Tuple, Optional, Union, Generator

import numpy as np
import pandas as pd
import rasterio
from rasterio.windows import Window
from shapely.geometry import Polygon, MultiPolygon

warnings.filterwarnings("ignore", category=rasterio.errors.NotGeoreferencedWarning)


# ----------------------------------------------------------------------------------------------------------------------
# Classes
# ----------------------------------------------------------------------------------------------------------------------


@dataclass
class TileConfig:
    def __init__(self,
                 slice_wh: Union[int, Tuple[int, int]],
                 overlap_wh: Union[int, Tuple[float, float]] = 0,
                 annotation_type: str = "object_detection",
                 ext: str = ".png",
                 densify_factor: float = 0.5,
                 smoothing_tolerance: float = 0.1,
                 train_ratio: float = 0.8,
                 valid_ratio: float = 0.1,
                 test_ratio: float = 0.1,
                 margins: Union[float, Tuple[float, float, float, float]] = 0.0):
        """
        Args:
            margins: Either a single float (0-1) for uniform margins,
                    or tuple (left, top, right, bottom) of floats (0-1) or ints
        """
        self.slice_wh = slice_wh if isinstance(slice_wh, tuple) else (slice_wh, slice_wh)
        self.overlap_wh = overlap_wh
        self.annotation_type = annotation_type
        self.ext = ext
        self.densify_factor = densify_factor
        self.smoothing_tolerance = smoothing_tolerance
        self.train_ratio = train_ratio
        self.valid_ratio = valid_ratio
        self.test_ratio = test_ratio
        
        # Handle margins
        if isinstance(margins, (int, float)):
            self.margins = (margins, margins, margins, margins)
        else:
            self.margins = margins
        
        self._validate()

    def _validate(self):
        # Add to existing validation
        if isinstance(self.margins[0], float):
            if not all(0 <= m <= 1 for m in self.margins):
                raise ValueError("Float margins must be between 0 and 1")
        elif isinstance(self.margins[0], int):
            if not all(m >= 0 for m in self.margins):
                raise ValueError("Integer margins must be non-negative")
        else:
            raise ValueError("Margins must be int or float")

    def get_effective_area(self, image_width: int, image_height: int) -> Tuple[int, int, int, int]:
        """Calculate the effective area after applying margins"""
        left, top, right, bottom = self.margins
        
        if isinstance(left, float):
            x_min = int(image_width * left)
            y_min = int(image_height * top)
            x_max = int(image_width * (1 - right))
            y_max = int(image_height * (1 - bottom))
        else:
            x_min = left
            y_min = top
            x_max = image_width - right
            y_max = image_height - bottom
            
        return x_min, y_min, x_max, y_max
    

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
        """
        self.source = Path(source)
        self.target = Path(target)
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

    def _calculate_step_size(self, slice_size: int, overlap: Union[int, float]) -> int:
        """Calculate effective step size for tiling."""
        if isinstance(overlap, float):
            overlap = int(slice_size * overlap)
        return slice_size - overlap

    def _calculate_num_tiles(self, img_size: int, step_size: int) -> int:
        """Calculate number of tiles in one dimension."""
        return math.ceil((img_size - step_size) / step_size)

    def _generate_tile_positions(self, img_size: int, step_size: int) -> np.ndarray:
        """Generate tile positions using numpy for faster calculations."""
        return np.arange(0, img_size, step_size)

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
        step_w = self._calculate_step_size(slice_w, overlap_w)
        step_h = self._calculate_step_size(slice_h, overlap_h)

        # Calculate number of tiles in each dimension
        num_tiles_w = self._calculate_num_tiles(img_w, step_w)
        num_tiles_h = self._calculate_num_tiles(img_h, step_h)

        # Generate tile positions using numpy for faster calculations
        x_coords = self._generate_tile_positions(img_w, step_w)
        y_coords = self._generate_tile_positions(img_h, step_h)

        for y1 in y_coords:
            for x1 in x_coords:
                x2 = min(x1 + slice_w, img_w)
                y2 = min(y1 + slice_h, img_h)

                # Handle edge cases by shifting tiles
                if x2 == img_w and x2 != x1 + slice_w:
                    x1 = max(0, x2 - slice_w)
                if y2 == img_h and y2 != y1 + slice_h:
                    y1 = max(0, y2 - slice_h)

                yield x1, y1, x2, y2

    def _densify_line(self, coords: List[Tuple[float, float]], factor: float) -> List[Tuple[float, float]]:
        """Add points along line segments to increase resolution"""
        result = []
        for i in range(len(coords) - 1):
            p1, p2 = coords[i], coords[i + 1]
            dx = p2[0] - p1[0]
            dy = p2[1] - p1[1]
            segment_length = math.sqrt(dx * dx + dy * dy)
            steps = int(segment_length / factor)
            
            if steps > 1:
                for step in range(steps):
                    t = step / steps
                    x = p1[0] + t * dx
                    y = p1[1] + t * dy
                    result.append((x, y))
            else:
                result.append(p1)
                
        result.append(coords[-1])
        return result

    def _process_polygon(self, poly: Polygon) -> List[List[Tuple[float, float]]]:
        # Calculate densification distance based on polygon size
        perimeter = poly.length
        dense_distance = perimeter * self.config.densify_factor
        
        # Process exterior ring
        coords = list(poly.exterior.coords)[:-1]
        dense_coords = self._densify_line(coords, dense_distance)
        
        # Create simplified version for smoothing
        dense_poly = Polygon(dense_coords)
        smoothed = dense_poly.simplify(self.config.smoothing_tolerance, preserve_topology=True)
        
        result = [list(smoothed.exterior.coords)[:-1]]
        
        # Process interior rings (holes)
        for interior in poly.interiors:
            coords = list(interior.coords)[:-1]
            dense_coords = self._densify_line(coords, dense_distance)
            hole_poly = Polygon(dense_coords)
            smoothed_hole = hole_poly.simplify(self.config.smoothing_tolerance, preserve_topology=True)
            result.append(list(smoothed_hole.exterior.coords)[:-1])
            
        return result

    def _process_intersection(self, intersection: Union[Polygon, MultiPolygon]) -> List[List[Tuple[float, float]]]:
        """
        Process intersection geometry with improved quality.
        
        Args:
            intersection: Shapely geometry object
            
        Returns:
            List of coordinate lists (exterior + holes)
        """
        if isinstance(intersection, Polygon):
            return self._process_polygon(intersection)
        else:  # MultiPolygon
            all_coords = []
            # Process all polygons, not just the largest
            for poly in intersection.geoms:
                all_coords.extend(self._process_polygon(poly))
            return all_coords

    def _normalize_coordinates(self, 
                               coord_lists: List[List[Tuple[float, float]]], 
                               tile_bounds: Tuple[int, int, int, int]) -> str:
        """
        Normalize coordinates to [0,1] range relative to tile bounds.
        
        Args:
            coord_lists: List of coordinate lists (exterior + holes)
            tile_bounds: (x1, y1, x2, y2) of tile bounds
            
        Returns:
            Space-separated string of normalized coordinates
        """
        x1, y1, x2, y2 = tile_bounds
        tile_width = x2 - x1
        tile_height = y2 - y1
        
        normalized_parts = []
        for coords in coord_lists:
            normalized = []
            for x, y in coords:
                norm_x = (x - x1) / tile_width
                norm_y = (y - y1) / tile_height
                normalized.append(f"{norm_x:.6f} {norm_y:.6f}")
            normalized_parts.append(normalized)
        
        # Join all parts with special separator
        return " ".join([" ".join(part) for part in normalized_parts])

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
            folder: Subfolder name (train, valid, test)
        """
        # Read image and labels
        with rasterio.open(image_path) as src:
            width, height = src.width, src.height

            # Get effective area
            x_min, y_min, x_max, y_max = self.config.get_effective_area(width, height)
            effective_width = x_max - x_min
            effective_height = y_max - y_min

            # Process annotations
            boxes = []
            with open(label_path) as f:
                for line in f:
                    parts = line.strip().split()
                    class_id = int(parts[0])
                    
                    if self.config.annotation_type == "object_detection":
                        # Parse fixed format: class x y w h
                        x_center = float(parts[1]) * width
                        y_center = float(parts[2]) * height
                        box_w = float(parts[3]) * width
                        box_h = float(parts[4]) * height
                        
                        x1 = x_center - box_w / 2
                        y1 = y_center - box_h / 2
                        x2 = x_center + box_w / 2
                        y2 = y_center + box_h / 2
                        boxes.append((class_id, Polygon([(x1, y1), (x2, y1), (x2, y2), (x1, y2)])))
                    else:
                        # Parse variable length format: class x1 y1 x2 y2 ...
                        points = []
                        for i in range(1, len(parts), 2):
                            x = float(parts[i]) * width
                            y = float(parts[i + 1]) * height
                            points.append((x, y))
                        boxes.append((class_id, Polygon(points)))

            # Process each tile
            for tile_idx, (x1, y1, x2, y2) in enumerate(self._calculate_tile_positions((effective_width,
                                                                                        effective_height))):
                window = Window(x1 + x_min, y1 + y_min, x2 - x1, y2 - y1)
                tile_data = src.read(window=window)
                tile_polygon = Polygon([(x1 + x_min, y1 + y_min),
                                        (x2 + x_min, y1 + y_min),
                                        (x2 + x_min, y2 + y_min),
                                        (x1 + x_min, y2 + y_min)])
                tile_labels = []

                # Process annotations for this tile
                for box_class, box_polygon in boxes:
                    if tile_polygon.intersects(box_polygon):
                        intersection = tile_polygon.intersection(box_polygon)
                        
                        if self.config.annotation_type == "object_detection":
                            # Handle object detection
                            bbox = intersection.envelope
                            center = bbox.centroid
                            bbox_coords = bbox.exterior.coords.xy
                            new_width = (max(bbox_coords[0]) - min(bbox_coords[0])) / (x2 - x1)
                            new_height = (max(bbox_coords[1]) - min(bbox_coords[1])) / (y2 - y1)
                            new_x = (center.x - x1) / (x2 - x1)
                            new_y = (center.y - y1) / (y2 - y1)
                            tile_labels.append([box_class, new_x, new_y, new_width, new_height])
                        else:
                            # Handle instance segmentation with improved processing
                            coord_lists = self._process_intersection(intersection)
                            normalized = self._normalize_coordinates(coord_lists, (x1, y1, x2, y2))
                            tile_labels.append([box_class, normalized])

                # Save tile image and labels
                tile_suffix = f'_tile_{tile_idx}{self.config.ext}'
                self._save_tile(tile_data, image_path, tile_suffix, tile_labels, folder)

    def _save_tile_image(self, tile_data: np.ndarray, image_path: Path, suffix: str, folder: str) -> None:
        """
        Save a tile image to the appropriate directory.

        Args:
            tile_data: Numpy array of tile image
            image_path: Path to original image
            suffix: Suffix for the tile filename
            folder: Subfolder name (train, valid, test)
        """
        # Set the save directory
        save_dir = self.target / folder

        # Save the image
        image_path = save_dir / "images" / image_path.name.replace(self.config.ext, suffix)
        with rasterio.open(
            image_path,
            'w',
            driver='GTiff',
            height=tile_data.shape[1],
            width=tile_data.shape[2],
            count=tile_data.shape[0],
            dtype=tile_data.dtype
        ) as dst:
            dst.write(tile_data)
        self.logger.info(f"Saved tile to {image_path}")

    def _save_tile_labels(self, labels: Optional[List], image_path: Path, suffix: str, folder: str) -> None:
        """
        Save tile labels to the appropriate directory.

        Args:
            labels: List of labels for the tile
            image_path: Path to original image
            suffix: Suffix for the tile filename
            folder: Subfolder name (train, valid, test)
        """
        if labels:
            # Save the labels in the appropriate directory
            label_path = self.target / folder / "labels" / image_path.name.replace(self.config.ext, suffix)
            label_path = label_path.with_suffix('.txt')
            is_segmentation = self.config.annotation_type == "instance_segmentation"
            self._save_labels(labels, label_path, is_segmentation)

    def _save_tile(self,
                   tile_data: np.ndarray, 
                   original_path: Path, 
                   suffix: str, 
                   labels: Optional[List],
                   folder: str) -> None:
        """
        Save a tile image and its labels.
        
        Args:
            tile_data: Numpy array of tile image
            original_path: Path to original image
            suffix: Suffix for the tile filename
            labels: List of labels for the tile
            folder: Subfolder name (train, valid, test)
        """
        self._save_tile_image(tile_data, original_path, suffix, folder)
        self._save_tile_labels(labels, original_path, suffix, folder)

    def split_data(self) -> None:
        """
        Split train data into train, valid, and test sets using specified ratios.
        Files are moved from train to valid/test directories.
        """
        train_images = list((self.target / 'train' / 'images').glob(f'*{self.config.ext}'))
        train_labels = list((self.target / 'train' / 'labels').glob('*.txt'))

        if not train_images or not train_labels:
            self.logger.warning("No train data found to split")
            return

        combined = list(zip(train_images, train_labels))
        random.shuffle(combined)
        train_images, train_labels = zip(*combined)

        num_train = int(len(train_images) * self.config.train_ratio)
        num_valid = int(len(train_images) * self.config.valid_ratio)

        valid_set = combined[num_train:num_train + num_valid]
        test_set = combined[num_train + num_valid:]

        # Move files to valid folder
        for image_path, label_path in valid_set:
            self._move_split_data(image_path, label_path, 'valid')

        # Move files to test folder
        for image_path, label_path in test_set:
            self._move_split_data(image_path, label_path, 'test')

    def _move_split_data(self, image_path: Path, label_path: Path, folder: str) -> None:
        """
        Move split data to the appropriate folder.

        Args:
            image_path: Path to image file
            label_path: Path to label file
            folder: Subfolder name (valid or test)
        """
        target_image = self.target / folder / "images" / image_path.name
        target_label = self.target / folder / "labels" / label_path.name

        image_path.rename(target_image)
        label_path.rename(target_label)

    def _validate_directories(self) -> None:
        """Validate source and target directories."""
        self._validate_yolo_structure(self.source)
        self._create_target_folder(self.target)

    def _process_subfolder(self, subfolder: str) -> None:
        """Process images and labels in a subfolder."""
        image_paths = list((self.source / subfolder / 'images').glob(f'*{self.config.ext}'))
        label_paths = list((self.source / subfolder / 'labels').glob('*.txt'))
            
        # Log the number of images, labels found
        self.logger.info(f'Found {len(image_paths)} images in {subfolder} directory')
        self.logger.info(f'Found {len(label_paths)} label files in {subfolder} directory')

        # Check for missing files
        if not image_paths:
            self.logger.warning(f"No images found in {subfolder} directory, skipping")
            return
        if len(image_paths) != len(label_paths):
            self.logger.error(f"Number of images and labels do not match in {subfolder} directory, skipping")
            return

        # Process each image
        for image_path, label_path in list(zip(image_paths, label_paths)):
            assert image_path.stem == label_path.stem, "Image and label filenames do not match"
            self.logger.info(f'Processing {image_path}')
            self.tile_image(image_path, label_path, subfolder)

    def _check_and_split_data(self) -> None:
        """Check if valid or test folders are empty and split data if necessary."""
        valid_images = list((self.target / 'valid' / 'images').glob(f'*{self.config.ext}'))
        test_images = list((self.target / 'test' / 'images').glob(f'*{self.config.ext}'))

        if not valid_images or not test_images:
            self.split_data()
            self.logger.info('Split train data into valid and test sets')

    def _copy_data_yaml(self) -> None:
        """Copy data.yaml from source to target directory if it exists."""
        data_yaml = self.source / 'data.yaml'
        if data_yaml.exists():
            copyfile(data_yaml, self.target / 'data.yaml')
        else:
            self.logger.warning('data.yaml not found in source directory')

    def run(self) -> None:
        """Run the complete tiling process"""
        try:
            # Validate directories
            self._validate_directories()

            # Train, valid, test subfolders
            for subfolder in self.subfolders:
                self._process_subfolder(subfolder)

            self.logger.info('Tiling process completed successfully')
            
            # Check if valid or test folders are empty
            self._check_and_split_data()
            
            # Copy data.yaml
            self._copy_data_yaml()

        except Exception as e:
            self.logger.error(f'Error during tiling process: {str(e)}')
            raise


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Tile YOLO dataset images and annotations.")

    parser.add_argument("source", type=str,
                        help="Source directory containing YOLO dataset")

    parser.add_argument("target", type=str,
                        help="Target directory for sliced dataset")

    parser.add_argument("--slice_wh", type=int, nargs=2, default=(640, 480),
                        help="Slice width and height")

    parser.add_argument("--overlap_wh", type=float, nargs=2, default=(0.1, 0.1),
                        help="Overlap width and height")

    parser.add_argument("--ext", type=str, default=".png",
                        help="Image extension")

    parser.add_argument("--annotation_type", type=str, default="object_detection",
                        help="Type of annotation")

    parser.add_argument("--densify_factor", type=float, default=0.01,
                        help="Densify factor for segmentation")

    parser.add_argument("--smoothing_tolerance", type=float, default=0.99,
                        help="Smoothing tolerance for segmentation")

    parser.add_argument("--train_ratio", type=float, default=0.7,
                        help="Train split ratio")

    parser.add_argument("--valid_ratio", type=float, default=0.2,
                        help="Validation split ratio")

    parser.add_argument("--test_ratio", type=float, default=0.1,
                        help="Test split ratio")

    return parser.parse_args()


def main():
    """Main function to run the tiling process based on command line arguments."""
    args = parse_args()
    config = TileConfig(
        slice_wh=tuple(args.slice_wh),
        overlap_wh=tuple(args.overlap_wh),
        ext=args.ext,
        annotation_type=args.annotation_type,
        densify_factor=args.densify_factor,
        smoothing_tolerance=args.smoothing_tolerance,
        train_ratio=args.train_ratio,
        valid_ratio=args.valid_ratio,
        test_ratio=args.test_ratio
    )
    tiler = YoloTiler(
        source=args.source,
        target=args.target,
        config=config,
    )
    tiler.run()


if __name__ == "__main__":
    main()