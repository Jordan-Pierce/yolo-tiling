import logging
import math
from dataclasses import dataclass
from pathlib import Path
from shutil import copyfile
from typing import List, Tuple, Optional, Union, Generator

import numpy as np
import pandas as pd
from PIL import Image
from shapely.geometry import Polygon, MultiPolygon
from tqdm import tqdm


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
    # densify factor for segmentation (smaller = more points)
    densify_factor: float = 0.01
    # smoothing tolerance for segmentation (smaller = less smoothing)
    smoothing_tolerance: float = 0.99

    def __post_init__(self):
        """Validate configuration parameters"""
        if not all(x > 0 for x in self.slice_wh):
            raise ValueError("Slice dimensions must be positive")
        if not all(x >= 0 for x in self.overlap_wh):
            raise ValueError("Overlap must be non-negative")
        if not all(o < s for o, s in zip(self.overlap_wh, self.slice_wh)):
            raise ValueError("Overlap must be less than slice size")
        if self.densify_factor <= 0 or self.densify_factor >= 1:
            raise ValueError("Densify factor must be between 0 and 1")
        if self.smoothing_tolerance <= 0 or self.smoothing_tolerance >= 1:
            raise ValueError("Smoothing tolerance must be between 0 and 1")


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

        # Generate tile positions using numpy for faster calculations
        x_coords = np.arange(0, img_w, step_w)
        y_coords = np.arange(0, img_h, step_h)

        for y1 in y_coords:
            for x1 in x_coords:
                x2 = min(x1 + slice_w, img_w)
                y2 = min(y1 + slice_h, img_h)

                # Handle edge cases by shifting tiles
                if x2 == img_w and x2 != x1 + slice_w:
                    x1 = max(0, x2 - slice_w)
                if y2 == img_h and y2 != y1 + slice_h:
                    y1 = max(0, y2 - slice_h)

                yield (x1, y1, x2, y2)

    def _process_intersection(self, intersection: Union[Polygon, MultiPolygon]) -> List[List[Tuple[float, float]]]:
        """
        Process intersection geometry with improved quality.
        
        Args:
            intersection: Shapely geometry object
            
        Returns:
            List of coordinate lists (exterior + holes)
        """
        def densify_line(coords: List[Tuple[float, float]], factor: float) -> List[Tuple[float, float]]:
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

        def process_polygon(poly: Polygon) -> List[List[Tuple[float, float]]]:
            # Calculate densification distance based on polygon size
            perimeter = poly.length
            dense_distance = perimeter * self.config.densify_factor
            
            # Process exterior ring
            coords = list(poly.exterior.coords)[:-1]
            dense_coords = densify_line(coords, dense_distance)
            
            # Create simplified version for smoothing
            dense_poly = Polygon(dense_coords)
            smoothed = dense_poly.simplify(self.config.smoothing_tolerance, preserve_topology=True)
            
            result = [list(smoothed.exterior.coords)[:-1]]
            
            # Process interior rings (holes)
            for interior in poly.interiors:
                coords = list(interior.coords)[:-1]
                dense_coords = densify_line(coords, dense_distance)
                hole_poly = Polygon(dense_coords)
                smoothed_hole = hole_poly.simplify(self.config.smoothing_tolerance, preserve_topology=True)
                result.append(list(smoothed_hole.exterior.coords)[:-1])
                
            return result

        if isinstance(intersection, Polygon):
            return process_polygon(intersection)
        else:  # MultiPolygon
            all_coords = []
            # Process all polygons, not just the largest
            for poly in intersection.geoms:
                all_coords.extend(process_polygon(poly))
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
        image = Image.open(image_path)
        image_array = np.array(image, dtype=np.uint8)
        width, height = image.size

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
                        # Handle instance segmentation with improved processing
                        coord_lists = self._process_intersection(intersection)
                        normalized = self._normalize_coordinates(coord_lists, (x1, y1, x2, y2))
                        tile_labels.append([box_class, normalized])
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

            # Save tile and annotations if it contains objects
            if has_annotations or not self.config.annotation_type == "instance_segmentation":
                tile_suffix = f'_tile_{tile_idx}{self.config.ext}'
                self._save_tile(tile_data, image_path, tile_suffix, tile_labels, folder)

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
        # Set the save directory
        save_dir = self.target / folder

        # Save the image
        image_path = save_dir / "images" / original_path.name.replace(self.config.ext, suffix)
        Image.fromarray(tile_data).save(image_path)
        self.logger.info(f"Saved tile image to {image_path}")
        
        if labels:
            # Save the labels in the appropriate directory
            label_path = save_dir / "labels" / original_path.name.replace(self.config.ext, suffix)
            label_path = label_path.with_suffix('.txt')
            is_segmentation = self.config.annotation_type == "instance_segmentation"
            if is_segmentation:
                with open(label_path, 'w') as f:
                    for label_class, points in labels:
                        f.write(f"{label_class} {points.replace(',', ' ')}\n")
            else:
                df = pd.DataFrame(labels, columns=['class', 'x1', 'y1', 'w', 'h'])
                df.to_csv(label_path, sep=' ', index=False, header=False, float_format='%.6f')
            self.logger.info(f"Saved tile labels to {label_path}")

    def _save_tile_image(self, tile_array: np.ndarray, original_path: Path, i: int, j: int) -> None:
        """
        Save a tile image to the appropriate directory.

        Args:
            tile_array: Numpy array of tile image
            original_path: Path to original image
            i, j: Tile indices
        """
        # Set the save directory
        save_path = self.target / original_path.name.replace(self.config.ext, f'_{i}_{j}{self.config.ext}')
        # Save the image
        Image.fromarray(tile_array).save(save_path)
        self.logger.info(f"Saved tile to {save_path}")

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
                    self.logger.warning(f"No images found in {subfolder} directory, skipping")
                    continue
                if len(image_paths) != len(label_paths):
                    self.logger.error(f"Number of images and labels do not match in {subfolder} directory, skipping")
                    continue

                # Process each image
                for image_path, label_path in tqdm(list(zip(image_paths, label_paths)), desc=f"Processing {subfolder} images"):
                    assert image_path.stem == label_path.stem, "Image and label filenames do not match"
                    self.logger.info(f'Processing {image_path}')
                    self.tile_image(image_path, label_path, subfolder)

            self.logger.info('Tiling process completed successfully')
            
            # Copy data.yaml
            data_yaml = self.source / 'data.yaml'
            if data_yaml.exists():
                copyfile(data_yaml, self.target / 'data.yaml')
            else:
                self.logger.warning('data.yaml not found in source directory')

        except Exception as e:
            self.logger.error(f'Error during tiling process: {str(e)}')
            raise
