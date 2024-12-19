import pandas as pd
import numpy as np
from PIL import Image
from shapely.geometry import Polygon
import glob
import argparse
import os
import random
from shutil import copyfile

class YoloTiler:
    def __init__(self, source, target, ext, falsefolder, size, ratio):
        self.source = source
        self.target = target
        self.ext = ext
        self.falsefolder = falsefolder
        self.size = size
        self.ratio = ratio

    def tiler(self, imnames, newpath, falsepath, slice_size, ext):
        for imname in imnames:
            im = Image.open(imname)
            imr = np.array(im, dtype=np.uint8)
            height = imr.shape[0]
            width = imr.shape[1]
            labname = imname.replace(ext, '.txt')
            labels = pd.read_csv(labname, sep=' ', names=['class', 'x1', 'y1', 'w', 'h'])

            labels[['x1', 'w']] = labels[['x1', 'w']] * width
            labels[['y1', 'h']] = labels[['y1', 'h']] * height

            boxes = []

            for row in labels.iterrows():
                x1 = row[1]['x1'] - row[1]['w']/2
                y1 = (height - row[1]['y1']) - row[1]['h']/2
                x2 = row[1]['x1'] + row[1]['w']/2
                y2 = (height - row[1]['y1']) + row[1]['h']/2

                boxes.append((int(row[1]['class']), Polygon([(x1, y1), (x2, y1), (x2, y2), (x1, y2)])))

            counter = 0
            print('Image:', imname)
            for i in range((height // slice_size)):
                for j in range((width // slice_size)):
                    x1 = j*slice_size
                    y1 = height - (i*slice_size)
                    x2 = ((j+1)*slice_size) - 1
                    y2 = (height - (i+1)*slice_size) + 1

                    pol = Polygon([(x1, y1), (x2, y1), (x2, y2), (x1, y2)])
                    imsaved = False
                    slice_labels = []

                    for box in boxes:
                        if pol.intersects(box[1]):
                            inter = pol.intersection(box[1])

                            if not imsaved:
                                sliced = imr[i*slice_size:(i+1)*slice_size, j*slice_size:(j+1)*slice_size]
                                sliced_im = Image.fromarray(sliced)
                                filename = imname.split('/')[-1]
                                slice_path = newpath + "/" + filename.replace(ext, f'_{i}_{j}{ext}')
                                slice_labels_path = newpath + "/" + filename.replace(ext, f'_{i}_{j}.txt')
                                print(slice_path)
                                sliced_im.save(slice_path)
                                imsaved = True

                            new_box = inter.envelope

                            centre = new_box.centroid

                            x, y = new_box.exterior.coords.xy

                            new_width = (max(x) - min(x)) / slice_size
                            new_height = (max(y) - min(y)) / slice_size

                            new_x = (centre.coords.xy[0][0] - x1) / slice_size
                            new_y = (y1 - centre.coords.xy[1][0]) / slice_size

                            counter += 1

                            slice_labels.append([box[0], new_x, new_y, new_width, new_height])

                    if len(slice_labels) > 0:
                        slice_df = pd.DataFrame(slice_labels, columns=['class', 'x1', 'y1', 'w', 'h'])
                        print(slice_df)
                        slice_df.to_csv(slice_labels_path, sep=' ', index=False, header=False, float_format='%.6f')

                    if not imsaved and falsepath:
                        sliced = imr[i*slice_size:(i+1)*slice_size, j*slice_size:(j+1)*slice_size]
                        sliced_im = Image.fromarray(sliced)
                        filename = imname.split('/')[-1]
                        slice_path = falsepath + "/" + filename.replace(ext, f'_{i}_{j}{ext}')

                        sliced_im.save(slice_path)
                        print('Slice without boxes saved')
                        imsaved = True

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

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("-source", default="./yolosample/ts/", help = "Source folder with images and labels needed to be tiled")
    parser.add_argument("-target", default="./yolosliced/ts/", help = "Target folder for a new sliced dataset")
    parser.add_argument("-ext", default=".JPG", help = "Image extension in a dataset. Default: .JPG")
    parser.add_argument("-falsefolder", default=None, help = "Folder for tiles without bounding boxes")
    parser.add_argument("-size", type=int, default=416, help = "Size of a tile. Dafault: 416")
    parser.add_argument("-ratio", type=float, default=0.8, help = "Train/test split ratio. Dafault: 0.8")

    args = parser.parse_args()

    yolo_tiler = YoloTiler(args.source, args.target, args.ext, args.falsefolder, args.size, args.ratio)
    yolo_tiler.run()
