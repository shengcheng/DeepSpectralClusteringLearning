import os
import scipy.io as sio
from PIL import Image
# from torch.utils.data import Dataset
from .Dataset import Dataset
import numpy as np
from tqdm import tqdm
import pandas as pd
from PIL import ImageFile
from utils import utils
import os.path as osp
from collections import defaultdict
from .PreProcessImage import PreProcessIm
ImageFile.LOAD_TRUNCATED_IMAGES = True


class CarsDataset(Dataset):
    def __init__(self, data_dir, ids_per_batch, ims_per_id, is_train, resize_h_w, crop_h_w, mirror, im_mean, im_std, scale):
        self.root_dir = data_dir
        self.ids_per_batch = ids_per_batch
        self.ims_per_id = ims_per_id
        self.is_train = is_train
        self.classes = sio.loadmat(os.path.join(self.root_dir, 'devkit/cars_meta.mat'))['class_names'][0]
        self.index_train_table = sio.loadmat(
            os.path.join(self.root_dir, 'devkit/cars_train_annos.mat')
        )['annotations'][0]
        self.index_test_table = sio.loadmat(
            os.path.join(self.root_dir, 'devkit/cars_test_annos_withlabels.mat')
        )['annotations'][0]
        if is_train:
            self.image_dir = osp.join(self.root_dir, 'car_train_cropped')
        else:
            self.image_dir = osp.join(self.root_dir, 'car_test_cropped')

        self.label_train_table = [(row[4][0][0], row[5][0]) for row in self.index_train_table]
        self.label_test_table = [(row[4][0][0], row[5][0]) for row in self.index_test_table]
        # self.im_names = [row[5][0] for row in self.index_table]
        # self.labels = [int(row[4][0][0]) for row in self.index_table]

        self.inds_table = defaultdict(list)
        self.im_names = []
        self.labels = []

        ind = 0
        for _, (id, im_name) in enumerate(self.label_train_table):
            if id <= 32 and is_train:
                self.im_names.append('train_' + im_name)
                self.labels.append(int(id))
                self.inds_table[int(id)].append(ind)
                ind += 1
            if id > 32 and not is_train:
                self.im_names.append('train_' + im_name)
                self.labels.append(int(id))
                self.inds_table[int(id)].append(ind)
                ind += 1

        for _, (id, im_name) in enumerate(self.label_test_table):
            if id <= 32 and is_train:
                self.im_names.append('test_' + im_name)
                self.labels.append(int(id))
                self.inds_table[int(id)].append(ind)
                ind += 1
            if id > 32 and not is_train:
                self.im_names.append('test_' + im_name)
                self.labels.append(int(id))
                self.inds_table[int(id)].append(ind)
                ind += 1

        self.ids = self.inds_table.keys()
        if is_train:
            dataset_len = len(self.ids)
            batch_size = self.ids_per_batch
            final_batch = False
        else:
            dataset_len = len(self.im_names)
            batch_size = self.ids_per_batch*self.ims_per_id
            final_batch = True
        super(CarsDataset, self).__init__(
            dataset_size=dataset_len,
            batch_size=batch_size,
            final_batch=final_batch
        )
        self.pre_process_im = PreProcessIm(resize_h_w, crop_h_w, mirror, im_mean, im_std, scale)

    def get_sample(self, ptr):
        if self.is_train:
            inds = self.inds_table[self.ids[ptr]]
            # inds = self.inds_table[ptr]
            if len(inds) < self.ims_per_id:
                inds = np.random.choice(inds, self.ims_per_id, replace=True)
            else:
                inds = np.random.choice(inds, self.ims_per_id, replace=False)
        else:
            inds = ptr
        im_names = [self.im_names[ind] for ind in inds]
        ims = [np.asarray(Image.open(osp.join(self.image_dir, name)))
               for name in im_names]
        ims = [self.pre_process_im(im) for im in ims]
        labels = np.asarray(self.labels)[inds]
        return ims, labels, im_names

    def next_batch(self):
        if self.epoch_done and self.shuffle:
            np.random.shuffle(self.ids)
        samples, self.epoch_done = self.prefetcher.next_batch()
        ims, labels, im_names = zip(*samples)
        ims = np.squeeze(np.stack(np.concatenate(ims)))
        im_names = np.concatenate(im_names)
        labels = np.concatenate(labels)
        return ims, im_names, labels, self.epoch_done





# class CarsDataset(Dataset):
#     def __init__(self, root_dir=None, train=None, transform=None):
#         self.root_dir = root_dir
#         self.train = train
#         self.transform = transform
#         self.train_count = 8144
#         self.test_count = 8041
#         if self.train:
#             self.offset = 0
#             self.image_dir = osp.join(self.root_dir, 'car_train_cropped')
#             self.index_table = sio.loadmat(
#                 osp.join(self.root_dir, 'devkit/cars_train_annos.mat')
#             )['annotations'][0]
#         else:
#             self.offset = self.train_count
#             self.image_dir = osp.join(self.root_dir, 'car_test_cropped')
#             self.index_table = sio.loadmat(
#                 osp.join(self.root_dir, 'devkit/cars_test_annos_withlabels.mat')
#             )['annotations'][0]
#         # self.image_dir = os.path.join(self.root_dir, 'car_ims_cropped')
#         # self.index_table = sio.loadmat(
#         #     os.path.join(self.root_dir, 'cars_annos.mat')
#         # )['annotations'][0]
#         # label table : list of ( label, directory ).
#         self.label_table = [(row[4][0][0], row[5][0]) for row in self.index_table]
#         # call class by self.classes[index][0]
#         self.classes = sio.loadmat(os.path.join(self.root_dir, 'devkit/cars_meta.mat'))['class_names'][0]
#         self.num_cluster = self.classes.size
#         # if not os.path.exists(self.image_dir):
#         #     self.crop_images()
#
#
#     def __len__(self):
#         if self.train:
#             return self.train_count
#         else:
#             return self.test_count
#
#     def __getitem__(self, idx):
#         # if not self.train:
#         #     idx += self.train_count
#         sample = dict()
#         sample['img'] = Image.open(os.path.join(self.image_dir, self.label_table[idx][1].split('/')[-1]))
#         sample['label'] = int(self.label_table[idx][0])
#         if self.transform is not None:
#             sample['img'] = self.transform(sample['img'])
#         return sample

