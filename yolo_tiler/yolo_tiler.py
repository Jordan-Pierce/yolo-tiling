import logging
import math
import random
import warnings
from tqdm import tqdm

from dataclasses import dataclass
from pathlib import Path
from shutil import copyfile
from typing import List, Tuple, Optional, Union, Generator, Callable

import cv2
import shapely
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio

from matplotlib.patches import Polygon as MplPolygon
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
                 input_ext: str = ".png",
                 output_ext: Optional[str] = None,
                 densify_factor: float = 0.5,
                 smoothing_tolerance: float = 0.1,
                 train_ratio: float = 0.8,
                 valid_ratio: float = 0.1,
                 test_ratio: float = 0.1,
                 margins: Union[float, Tuple[float, float, float, float]] = 0.0,
                 include_negative_samples: bool = True):
        """
        Args:
            margins: Either a single float (0-1) for uniform margins,
                    or tuple (left, top, right, bottom) of floats (0-1) or ints
            include_negative_samples: Boolean to determine if negative samples should be included
        """
        self.slice_wh = slice_wh if isinstance(slice_wh, tuple) else (slice_wh, slice_wh)
        self.overlap_wh = overlap_wh
        self.annotation_type = annotation_type
        self.input_ext = input_ext
        self.output_ext = output_ext if output_ext is not None else input_ext
        self.densify_factor = densify_factor
        self.smoothing_tolerance = smoothing_tolerance
        self.train_ratio = train_ratio
        self.valid_ratio = valid_ratio
        self.test_ratio = test_ratio
        self.include_negative_samples = include_negative_samples

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


@dataclass
class TileProgress:
    """Data class to track tiling progress"""
    current_tile: int
    total_tiles: int
    current_set: str
    current_image: str


class YoloTiler:
    """
    A class to tile YOLO dataset images and their corresponding annotations.
    Supports both object detection and instance segmentation formats.
    """

    def __init__(self,
                 source: Union[str, Path],
                 target: Union[str, Path],
                 config: TileConfig,
                 num_viz_samples: int = 0,
                 show_processing_status: bool = True,
                 progress_callback: Optional[Callable[[TileProgress], None]] = None):
        """
        Initialize YoloTiler with source and target directories.

        Args:
            source: Source directory containing YOLO dataset
            target: Target directory for sliced dataset
            config: TileConfig object containing tiling parameters
            num_viz_samples: Number of random samples to visualize from train set
            show_processing_status: Whether to show processing status
            progress_callback: Optional callback function to report progress
        """
        self.source = Path(source)
        self.target = Path(target)
        self.config = config
        self.num_viz_samples = num_viz_samples
        
        # Add show_process_status parameter and initialize progress bars dict
        self.show_process_status = show_processing_status
        self._progress_bars = {}
        
        # Set up the progress callback based on parameters
        if progress_callback is not None:
            self.progress_callback = progress_callback
        elif show_processing_status:
            self.progress_callback = self._tqdm_callback
        else:
            self.progress_callback = None
        
        self.logger = self._setup_logger()
        self.subfolders = ['train/', 'valid/', 'test/']

        # Create rendered directory if visualization is requested
        if self.num_viz_samples > 0:
            self.render_dir = self.target / 'rendered'
            self.render_dir.mkdir(parents=True, exist_ok=True)
            
    def _tqdm_callback(self, progress: TileProgress):
        """Internal callback function that uses tqdm for progress tracking
        
        Args:
            progress: TileProgress object containing current progress
            
        """
        if progress.current_set not in self._progress_bars:
            self._progress_bars[progress.current_set] = tqdm(
                total=progress.total_tiles,
                desc=progress.current_set,
                unit='items'
            )
        
        # Update progress
        self._progress_bars[progress.current_set].n = progress.current_tile
        self._progress_bars[progress.current_set].refresh()
        
        # Close and cleanup if task is complete
        if progress.current_tile >= progress.total_tiles:
            self._progress_bars[progress.current_set].close()
            del self._progress_bars[progress.current_set]

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

    def _count_total_tiles(self, image_size: Tuple[int, int]) -> int:
        """Count total number of tiles for an image"""
        img_w, img_h = image_size
        slice_w, slice_h = self.config.slice_wh
        overlap_w, overlap_h = self.config.overlap_wh

        # Calculate effective step sizes
        step_w = self._calculate_step_size(slice_w, overlap_w)
        step_h = self._calculate_step_size(slice_h, overlap_h)

        # Generate tile positions using numpy for faster calculations
        x_coords = self._generate_tile_positions(img_w, step_w)
        y_coords = self._generate_tile_positions(img_h, step_h)

        return len(x_coords) * len(y_coords)

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
        Calculate tile positions with overlap, respecting margins.

        Args:
            image_size: (width, height) of the image after margins applied

        Yields:
            Tuples of (x1, y1, x2, y2) for each tile within effective area
        """
        img_w, img_h = image_size
        slice_w, slice_h = self.config.slice_wh
        overlap_w, overlap_h = self.config.overlap_wh

        # Calculate effective step sizes
        step_w = self._calculate_step_size(slice_w, overlap_w)
        step_h = self._calculate_step_size(slice_h, overlap_h)

        # Generate tile positions using numpy for faster calculations
        # Use effective dimensions (after margins)
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
        """Process intersection geometry with proper polygon closure."""
        from shapely.geometry import LineString, Polygon, MultiPolygon
        
        def process_single_polygon(geom) -> List[List[Tuple[float, float]]]:
            # Handle LineString case
            if isinstance(geom, LineString):
                # Convert LineString to a very thin polygon
                buffer_dist = 1e-10
                geom = geom.buffer(buffer_dist)
            
            if not isinstance(geom, Polygon):
                return []
                
            # Ensure proper closure of exterior ring
            exterior_coords = list(geom.exterior.coords)
            if exterior_coords[0] != exterior_coords[-1]:
                exterior_coords.append(exterior_coords[0])

            result = [exterior_coords[:-1]]  # Remove duplicate closing point

            # Process holes with proper closure
            for interior in geom.interiors:
                interior_coords = list(interior.coords)
                if interior_coords[0] != interior_coords[-1]:
                    interior_coords.append(interior_coords[0])
                result.append(interior_coords[:-1])

            return result

        if isinstance(intersection, MultiPolygon):
            all_coords = []
            for poly in intersection.geoms:
                all_coords.extend(process_single_polygon(poly))
            return all_coords
        else:
            return process_single_polygon(intersection)

            if isinstance(intersection, Polygon):
                return process_single_polygon(intersection)
            else:  # MultiPolygon
                all_coords = []
                for poly in intersection.geoms:
                    all_coords.extend(process_single_polygon(poly))
                return all_coords

    def _normalize_coordinates(self, 
                               coord_lists: List[List[Tuple[float, float]]],
                               tile_bounds: Tuple[int, int, int, int]) -> str:
        """Normalize coordinates with proper polygon closure."""
        x1, y1, x2, y2 = tile_bounds
        tile_width = x2 - x1
        tile_height = y2 - y1

        normalized_parts = []
        for coords in coord_lists:
            # Ensure proper closure
            if coords[0] != coords[-1]:
                coords = coords + [coords[0]]

            normalized = []
            for x, y in coords:
                norm_x = max(0, min(1, (x - x1) / tile_width))  # Clamp to [0,1]
                norm_y = max(0, min(1, (y - y1) / tile_height))
                normalized.append(f"{norm_x:.6f} {norm_y:.6f}")
            normalized_parts.append(normalized)

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
        Tile an image and its corresponding labels, properly handling margins.
        """
        def clean_geometry(geom: Polygon) -> Polygon:
            """Clean potentially invalid geometry"""
            if not geom.is_valid:
                # Apply small buffer to fix self-intersections
                cleaned = geom.buffer(0)
                if cleaned.is_valid:
                    return cleaned
                # Try more aggressive cleaning if needed
                return geom.buffer(1e-10).buffer(-1e-10)
            return geom

        # Read image and labels
        with rasterio.open(image_path) as src:
            width, height = src.width, src.height

            # Get effective area (area after margins applied)
            x_min, y_min, x_max, y_max = self.config.get_effective_area(width, height)
            effective_width = x_max - x_min
            effective_height = y_max - y_min

            # Create polygon representing effective area (excludes margins)
            effective_area = Polygon([
                (x_min, y_min),
                (x_max, y_min),
                (x_max, y_max),
                (x_min, y_max)
            ])

            # Calculate total tiles for progress tracking
            total_tiles = self._count_total_tiles((effective_width, effective_height))

            # Process annotations
            boxes = []
            with open(label_path) as f:
                for line in f:
                    try:
                        parts = line.strip().split()
                        class_id = int(parts[0])

                        if self.config.annotation_type == "object_detection":
                            # Parse normalized coordinates
                            x_center_norm = float(parts[1])
                            y_center_norm = float(parts[2])
                            box_w_norm = float(parts[3])
                            box_h_norm = float(parts[4])

                            # Convert to absolute coordinates
                            x_center = x_center_norm * width
                            y_center = y_center_norm * height
                            box_w = box_w_norm * width
                            box_h = box_h_norm * height

                            x1 = x_center - box_w / 2
                            y1 = y_center - box_h / 2
                            x2 = x_center + box_w / 2
                            y2 = y_center + box_h / 2
                            box_polygon = Polygon([(x1, y1), (x2, y1), (x2, y2), (x1, y2)])

                            # Only include if box intersects with effective area
                            if box_polygon.intersects(effective_area):
                                # Clip box to effective area
                                clipped_box = box_polygon.intersection(effective_area)
                                if not clipped_box.is_empty:
                                    boxes.append((class_id, clipped_box))
                        else:
                            # Instance segmentation
                            points = []
                            for i in range(1, len(parts), 2):
                                x_norm = float(parts[i])
                                y_norm = float(parts[i + 1])
                                x = x_norm * width
                                y = y_norm * height
                                points.append((x, y))

                            try:
                                polygon = Polygon(points)
                                # Clean and validate polygon
                                polygon = clean_geometry(polygon)
                                
                                if polygon.is_valid and polygon.intersects(effective_area):
                                    # Safely perform intersection
                                    try:
                                        clipped_polygon = polygon.intersection(effective_area)
                                        if not clipped_polygon.is_empty:
                                            boxes.append((class_id, clipped_polygon))
                                    except (shapely.errors.GEOSException, ValueError) as e:
                                        print(f"Warning: Failed to process polygon in {image_path.name}: {e}")
                                        continue
                            except Exception as e:
                                print(f"Warning: Invalid polygon in {image_path.name}: {e}")
                                continue
                    except Exception as e:
                        print(f"Warning: Failed to process line in {label_path}: {e}")
                        continue

            # Calculate tile positions
            effective_areas = self._calculate_tile_positions((effective_width, effective_height))
            
            # Process each tile within effective area
            for tile_idx, (x1, y1, x2, y2) in enumerate(effective_areas):
                # Convert tile coordinates to absolute image coordinates
                abs_x1 = x1 + x_min
                abs_y1 = y1 + y_min
                abs_x2 = x2 + x_min
                abs_y2 = y2 + y_min

                if self.progress_callback:
                    progress = TileProgress(
                        current_tile=tile_idx + 1,
                        total_tiles=total_tiles,
                        current_set=folder.rstrip('/'),
                        current_image=image_path.name
                    )
                    self.progress_callback(progress)

                # Extract tile data
                window = Window(abs_x1, abs_y1, abs_x2 - abs_x1, abs_y2 - abs_y1)
                tile_data = src.read(window=window)

                # Create polygon for current tile
                tile_polygon = Polygon([
                    (abs_x1, abs_y1),
                    (abs_x2, abs_y1),
                    (abs_x2, abs_y2),
                    (abs_x1, abs_y2)
                ])

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

                            # Normalize relative to tile dimensions
                            tile_width = abs_x2 - abs_x1
                            tile_height = abs_y2 - abs_y1

                            new_width = (max(bbox_coords[0]) - min(bbox_coords[0])) / tile_width
                            new_height = (max(bbox_coords[1]) - min(bbox_coords[1])) / tile_height
                            new_x = (center.x - abs_x1) / tile_width
                            new_y = (center.y - abs_y1) / tile_height

                            tile_labels.append([box_class, new_x, new_y, new_width, new_height])
                        else:
                            # Handle instance segmentation
                            coord_lists = self._process_intersection(intersection)
                            normalized = self._normalize_coordinates(coord_lists, (abs_x1, abs_y1, abs_x2, abs_y2))
                            tile_labels.append([box_class, normalized])

                # Save tile image and labels if include_negative_samples is True or there are labels
                if self.config.include_negative_samples or tile_labels:
                    tile_suffix = f'_tile_{tile_idx}{self.config.output_ext}'
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
        image_path = save_dir / "images" / image_path.name.replace(self.config.input_ext, suffix)
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
        # self.logger.info(f"Saved tile to {image_path}")

    def _save_tile_labels(self, labels: Optional[List], image_path: Path, suffix: str, folder: str) -> None:
        """
        Save tile labels to the appropriate directory.

        Args:
            labels: List of labels for the tile
            image_path: Path to original image
            suffix: Suffix for the tile filename
            folder: Subfolder name (train, valid, test)
        """
        # Save the labels in the appropriate directory
        label_path = self.target / folder / "labels" / image_path.name.replace(self.config.input_ext, suffix)
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
        train_images = list((self.target / 'train' / 'images').glob(f'*{self.config.output_ext}'))
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
        num_test = len(test_set)

        # Move files to valid folder
        for tile_idx, (image_path, label_path) in enumerate(valid_set):
            self._move_split_data(image_path, label_path, 'valid')

            if self.progress_callback:
                progress = TileProgress(
                    current_tile=tile_idx + 1,
                    total_tiles=num_valid,
                    current_set='valid',
                    current_image=image_path.name
                )
                self.progress_callback(progress)

        # Move files to test folder
        for tile_idx, (image_path, label_path) in enumerate(test_set):
            self._move_split_data(image_path, label_path, 'test')
            if self.progress_callback:
                progress = TileProgress(
                    current_tile=tile_idx + 1,
                    total_tiles=num_test,
                    current_set='test',
                    current_image=image_path.name
                )
                self.progress_callback(progress)

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
        image_paths = list((self.source / subfolder / 'images').glob(f'*{self.config.input_ext}'))
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
        valid_images = list((self.target / 'valid' / 'images').glob(f'*{self.config.output_ext}'))
        test_images = list((self.target / 'test' / 'images').glob(f'*{self.config.output_ext}'))

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

    def visualize_random_samples(self) -> None:
        """
        Visualize random samples from the train folder with their annotations.
        """
        if self.num_viz_samples <= 0:
            return

        # Get all image paths from train folder
        train_image_dir = self.target / 'train' / 'images'
        train_label_dir = self.target / 'train' / 'labels'

        image_paths = list(train_image_dir.glob(f'*{self.config.output_ext}'))

        if not image_paths:
            self.logger.warning("No images found in train folder for visualization")
            return

        # Select random samples
        num_samples = min(self.num_viz_samples, len(image_paths))
        selected_images = random.sample(image_paths, num_samples)

        # Process each selected image
        for tile_idx, image_path in enumerate(selected_images):
            label_path = train_label_dir / f"{image_path.stem}.txt"

            if not label_path.exists():
                self.logger.warning(f"Label file not found for {image_path.name}")
                continue

            if self.progress_callback:
                progress = TileProgress(
                    current_tile=tile_idx + 1,
                    total_tiles=num_samples,
                    current_set='rendered',
                    current_image=image_path.name
                )
                self.progress_callback(progress)

            self._render_single_sample(image_path, label_path, tile_idx + 1)

    def _render_single_sample(self, image_path: Path, label_path: Path, idx: int) -> None:
        """
        Render a single sample with its annotations.

        Args:
            image_path: Path to the image file
            label_path: Path to the label file
            idx: Index for the output filename
        """
        # Read image using OpenCV
        img = cv2.imread(str(image_path))
        if img is None:
            self.logger.warning(f"Could not read image: {image_path}")
            return

        # Convert BGR to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        height, width = img.shape[:2]

        # Create figure and axis
        fig, ax = plt.subplots(1)
        ax.imshow(img)

        # Random colors for different classes
        np.random.seed(42)  # For consistent colors
        colors = np.random.rand(100, 3)  # Support up to 100 classes

        # Read and parse labels
        with open(label_path) as f:
            for line in f:
                parts = line.strip().split()
                class_id = int(parts[0])
                color = colors[class_id % len(colors)]

                if self.config.annotation_type == "object_detection":
                    # Parse bounding box
                    x_center = float(parts[1]) * width
                    y_center = float(parts[2]) * height
                    box_w = float(parts[3]) * width
                    box_h = float(parts[4]) * height

                    # Calculate box coordinates
                    x = x_center - box_w / 2
                    y = y_center - box_h / 2

                    # Create rectangle patch with transparency
                    rect = patches.Rectangle(
                        (x, y),
                        box_w,
                        box_h,
                        linewidth=2,
                        edgecolor=color,
                        facecolor=color,
                        alpha=0.3  # Add transparency
                    )
                    ax.add_patch(rect)

                else:  # instance segmentation
                    # Parse polygon coordinates
                    coords = []
                    for i in range(1, len(parts), 2):
                        x = float(parts[i]) * width
                        y = float(parts[i + 1]) * height
                        coords.append([x, y])

                    # Create polygon patch with transparency
                    polygon = MplPolygon(
                        coords,
                        facecolor=color,
                        edgecolor=color,
                        linewidth=2,
                        alpha=0.3  # Add transparency
                    )
                    ax.add_patch(polygon)

        # Remove axes
        ax.axis('off')

        # Adjust layout
        plt.tight_layout()

        # Save the visualization
        output_path = self.render_dir / f"sample_{idx:03d}.jpg"
        plt.savefig(output_path, bbox_inches='tight', pad_inches=0, dpi=300)
        plt.close()

        # self.logger.info(f"Saved visualization to {output_path}")

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

            # Generate visualizations if requested
            if self.num_viz_samples > 0:
                self.logger.info(f'Generating {self.num_viz_samples} visualization samples...')
                self.visualize_random_samples()
                self.logger.info('Visualization generation completed')

        except Exception as e:
            self.logger.error(f'Error during tiling process: {str(e)}')
            raise
        
    def __del__(self):
        """Cleanup method to ensure all progress bars are closed"""
        for pbar in self._progress_bars.values():
            pbar.close()
        self._progress_bars.clear()
