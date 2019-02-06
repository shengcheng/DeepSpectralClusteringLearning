import os
import scipy.io as sio
from PIL import Image
import numpy as np
from tqdm import tqdm
import pandas as pd
from PIL import ImageFile
import shutil

import os.path as osp
ImageFile.LOAD_TRUNCATED_IMAGES = True

def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)
def mkdirs(paths):
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)

def crop_images(root_dir):
        # To generate crop car images.
        print("Cropping Images...")
        for type in ['train', 'test']:
            crop_image_train_dir = os.path.join(root_dir, 'car_train_cropped')
            crop_image_test_dir = os.path.join(root_dir, 'car_test_cropped')
            mkdir(crop_image_train_dir)
            mkdir(crop_image_test_dir)
            if type == 'train':
                image_dir = os.path.join(root_dir, 'cars_train')
                index_table = sio.loadmat(os.path.join(root_dir, 'devkit/cars_train_annos.mat'))['annotations'][0]
            else:
                image_dir = os.path.join(root_dir, 'cars_test')
                index_table = sio.loadmat(os.path.join(root_dir, 'devkit/cars_test_annos_withlabels.mat'))['annotations'][0]

            for row in tqdm(index_table):
                name = row[5][0]
                name = name.encode('ascii')

                id = row[4][0][0]

                image = Image.open(
                    os.path.join(image_dir, name)
                )
                image = image.crop(
                    (
                        row[0][0][0],
                        row[1][0][0],
                        row[2][0][0],
                        row[3][0][0],
                    )
                )
                if type == 'train':
                    name = 'train_' + name
                else:
                    name = 'test_' + name
                if id <= 98:
                    image.save(os.path.join(crop_image_train_dir, name.split('/')[-1]))
                else:
                    image.save(os.path.join(crop_image_test_dir, name.split('/')[-1]))

def move_dir(root_dir):
    image_dir = os.path.join(root_dir, 'car_test_cropped')
    dis_dir = os.path.join(root_dir, 'test')
    mkdir(dis_dir)
    index_table = sio.loadmat(os.path.join(root_dir, 'devkit/cars_test_annos_withlabels.mat'))['annotations'][0]
    label_table = [(row[4][0][0], row[5][0]) for row in index_table]
    for ind, (id, im_name) in enumerate(label_table):
        im_name = im_name.encode('ascii')
        image_path = os.path.join(image_dir, im_name)
        dis_path = os.path.join(dis_dir, im_name)
        if id > 96:
            shutil.move(image_path, dis_path)

if __name__ == '__main__':
    root_dir = '/home/cs/Desktop/REID/DeepSpectralClustering/cars'
    # crop_images(root_dir, True)
    # crop_images(root_dir, False)
    crop_images(root_dir)
