import pandas as pd
import numpy as np
from PIL import Image
from shapely.geometry import Polygon
import glob
import argparse
import os
import random
from shutil import copyfile


# ----------------------------------------------------------------------------------------------------------------------------------------
# Classes
# ----------------------------------------------------------------------------------------------------------------------------------------


class YoloTiler:
    def __init__(self, source, target, ext, falsefolder, size, ratio, annotation_type="object_detection"):
        self.source = source
        self.target = target
        self.ext = ext
        self.falsefolder = falsefolder
        self.size = size
        self.ratio = ratio
        self.annotation_type = annotation_type

    def tiler(self, imnames, newpath, falsepath, slice_size, ext):
        for imname in imnames:
            im = Image.open(imname)
            imr = np.array(im, dtype=np.uint8)
            height = imr.shape[0]
            width = imr.shape[1]
            labname = imname.replace(ext, '.txt')
            labels = pd.read_csv(labname, sep=' ', names=['class', 'x1', 'y1', 'w', 'h'] if self.annotation_type == "object_detection" else ['class', 'points'])

            boxes = []
            for row in labels.iterrows():
                if self.annotation_type == "object_detection":
                    # Original object detection code
                    x1 = row[1]['x1'] * width - (row[1]['w'] * width)/2
                    y1 = height - (row[1]['y1'] * height) - (row[1]['h'] * height)/2
                    x2 = row[1]['x1'] * width + (row[1]['w'] * width)/2
                    y2 = height - (row[1]['y1'] * height) + (row[1]['h'] * height)/2
                    boxes.append((int(row[1]['class']), Polygon([(x1, y1), (x2, y1), (x2, y2), (x1, y2)])))
                else:
                    # Instance segmentation code
                    points = row[1]['points'].split(';')
                    polygon_points = []
                    for point in points:
                        x, y = point.split(',')
                        x_coord = float(x) * width
                        y_coord = height - (float(y) * height)  # Convert to image coordinates
                        polygon_points.append((x_coord, y_coord))
                    boxes.append((int(row[1]['class']), Polygon(polygon_points)))

            counter = 0
            print('Image:', imname)
            for i in range((height // slice_size)):
                for j in range((width // slice_size)):
                    x1 = j*slice_size
                    y1 = height - (i*slice_size)
                    x2 = ((j+1)*slice_size) - 1
                    y2 = (height - (i+1)*slice_size) + 1

                    tile_polygon = Polygon([(x1, y1), (x2, y1), (x2, y2), (x1, y2)])
                    imsaved = False
                    slice_labels = []

                    for box in boxes:
                        if tile_polygon.intersects(box[1]):
                            inter = tile_polygon.intersection(box[1])

                            if not imsaved:
                                sliced = imr[i*slice_size:(i+1)*slice_size, j*slice_size:(j+1)*slice_size]
                                sliced_im = Image.fromarray(sliced)
                                filename = imname.split('/')[-1]
                                slice_path = newpath + "/" + filename.replace(ext, f'_{i}_{j}{ext}')
                                slice_labels_path = newpath + "/" + filename.replace(ext, f'_{i}_{j}.txt')
                                print(slice_path)
                                sliced_im.save(slice_path)
                                imsaved = True

                            if self.annotation_type == "object_detection":
                                # Original bounding box conversion code
                                new_box = inter.envelope
                                centre = new_box.centroid
                                x, y = new_box.exterior.coords.xy
                                new_width = (max(x) - min(x)) / slice_size
                                new_height = (max(y) - min(y)) / slice_size
                                new_x = (centre.coords.xy[0][0] - x1) / slice_size
                                new_y = (y1 - centre.coords.xy[1][0]) / slice_size
                                slice_labels.append([box[0], new_x, new_y, new_width, new_height])
                            else:
                                # Handle instance segmentation polygon
                                # Get intersection coordinates and normalize to tile coordinates
                                if isinstance(inter, Polygon):
                                    coords = list(inter.exterior.coords)[:-1]  # Remove the last point as it repeats the first
                                else:
                                    # Handle MultiPolygon case by taking the largest polygon
                                    largest_poly = max(inter.geoms, key=lambda p: p.area)
                                    coords = list(largest_poly.exterior.coords)[:-1]

                                # Convert coordinates to normalized format relative to tile
                                normalized_coords = []
                                for x, y in coords:
                                    norm_x = (x - x1) / slice_size  # Normalize X to [0,1]
                                    norm_y = (y1 - y) / slice_size  # Normalize Y to [0,1]
                                    normalized_coords.append(f"{norm_x:.6f},{norm_y:.6f}")
                                
                                # Join coordinates with semicolon for YOLO format
                                coord_string = ";".join(normalized_coords)
                                slice_labels.append([box[0], coord_string])

                            counter += 1

                    if len(slice_labels) > 0:
                        # Save labels based on annotation type
                        if self.annotation_type == "instance_segmentation":
                            # For instance segmentation, just save class and points
                            with open(slice_labels_path, 'w') as f:
                                for label in slice_labels:
                                    f.write(f"{label[0]} {label[1]}\n")
                        else:
                            # For object detection, save in standard YOLO format
                            slice_df = pd.DataFrame(slice_labels, columns=['class', 'x1', 'y1', 'w', 'h'])
                            slice_df.to_csv(slice_labels_path, sep=' ', index=False, header=False, float_format='%.6f')

                    if not imsaved and falsepath:
                        sliced = imr[i*slice_size:(i+1)*slice_size, j*slice_size:(j+1)*slice_size]
                        sliced_im = Image.fromarray(sliced)
                        filename = imname.split('/')[-1]
                        slice_path = falsepath + "/" + filename.replace(ext, f'_{i}_{j}{ext}')
                        sliced_im.save(slice_path)
                        print('Slice without boxes saved')
                        imsaved = True

    # The rest of the class methods (splitter, run) remain unchanged
    def splitter(self, target, target_upfolder, ext, ratio):
        imnames = glob.glob(f'{target}/*{ext}')
        names = [name.split('/')[-1] for name in imnames]

        train = []
        test = []
        for name in names:
            if random.random() > ratio:
                test.append(os.path.join(target, name))
            else:
                train.append(os.path.join(target, name))
        print('train:', len(train))
        print('test:', len(test))

        with open(f'{target_upfolder}/train.txt', 'w') as f:
            for item in train:
                f.write("%s\n" % item)

        with open(f'{target_upfolder}/test.txt', 'w') as f:
            for item in test:
                f.write("%s\n" % item)

    def run(self):
        imnames = glob.glob(f'{self.source}/*{self.ext}')
        labnames = glob.glob(f'{self.source}/*.txt')

        if len(imnames) == 0:
            raise Exception("Source folder should contain some images")
        elif len(imnames) != len(labnames):
            raise Exception("Dataset should contain equal number of images and txt files with labels")

        if not os.path.exists(self.target):
            os.makedirs(self.target)
        elif len(os.listdir(self.target)) > 0:
            raise Exception("Target folder should be empty")

        upfolder = os.path.join(self.source, '..' )
        target_upfolder = os.path.join(self.target, '..' )
        if not os.path.exists(os.path.join(upfolder, 'classes.names')):
            print('classes.names not found. It should be located one level higher than images')
        else:
            copyfile(os.path.join(upfolder, 'classes.names'), os.path.join(target_upfolder, 'classes.names'))

        if self.falsefolder:
            if not os.path.exists(self.falsefolder):
                os.makedirs(self.falsefolder)
            elif len(os.listdir(self.falsefolder)) > 0:
                raise Exception("Folder for tiles without boxes should be empty")

        self.tiler(imnames, self.target, self.falsefolder, self.size, self.ext)
        self.splitter(self.target, target_upfolder, self.ext, self.ratio)


# ----------------------------------------------------------------------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------------------------------------------------------------------


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--source", default="./yolosample/ts/", help = "Source folder with images and labels needed to be tiled")
    parser.add_argument("--target", default="./yolosliced/ts/", help = "Target folder for a new sliced dataset")
    parser.add_argument("--ext", default=".JPG", help = "Image extension in a dataset. Default: .JPG")
    parser.add_argument("--falsefolder", default=None, help = "Folder for tiles without bounding boxes")
    parser.add_argument("--size", type=int, default=416, help = "Size of a tile. Dafault: 416")
    parser.add_argument("--ratio", type=float, default=0.8, help = "Train/test split ratio. Dafault: 0.8")
    parser.add_argument("--annotation_type", default="object_detection", help = "Type of annotation: object_detection or instance_segmentation")

    args = parser.parse_args()

    yolo_tiler = YoloTiler(args.source, args.target, args.ext, args.falsefolder, args.size, args.ratio, args.annotation_type)
    yolo_tiler.run()
