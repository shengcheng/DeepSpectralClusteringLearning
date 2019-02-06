import cv2
import numpy as np


class PreProcessIm(object):
    def __init__(
            self,
            resize_h_w=None,
            crop_h_w=None,
            mirror=None,
            im_mean=None,
            im_std=None,
            scale=None,
            batch_dims='NCHW'):

        self.resize_h_w = resize_h_w
        self.crop_h_w = crop_h_w
        self.mirror = mirror
        self.im_mean = im_mean
        self.im_std = im_std
        self.scale = scale
        self.check_batch_dims(batch_dims)
        self.batch_dims = batch_dims
        self.prng = np.random

    def __call__(self, im):
        return self.pre_process_im(im)

    @staticmethod
    def check_mirror_type(mirror_type):
        assert mirror_type in [None, 'random', 'always']

    @staticmethod
    def check_batch_dims(batch_dims):
        # 'N': batch size, 'C': num channels, 'H': im height, 'W': im width
        # PyTorch uses 'NCHW', while TensorFlow uses 'NHWC'.
        assert batch_dims in ['NCHW', 'NHWC']

    def set_mirror_type(self, mirror_type):
        self.check_mirror_type(mirror_type)
        self.mirror_type = mirror_type

    @staticmethod
    def rand_crop_im(im, new_size, prng=np.random):
        """Crop `im` to `new_size`: [new_w, new_h]."""
        if (new_size[0] == im.shape[0]) and (new_size[1] == im.shape[1]):
            return im
        h_start = prng.randint(0, im.shape[0] - new_size[0])
        w_start = prng.randint(0, im.shape[1] - new_size[1])
        im = np.copy(
            im[h_start: h_start + new_size[0], w_start: w_start + new_size[1], :])
        return im

    @staticmethod
    def center_crop(im, new_size):
        h, w = im.shape
        th, tw = new_size.shape
        h_start = int(round((h - th) / 2.))
        w_start = int(round((w - tw) / 2.))
        return np.copy(
            im[h_start: h_start+th, w_start:w_start+tw,:]
        )

    @staticmethod
    def replicate_dim(im):
        im = np.tile(im, (3, 1, 1)).transpose(1, 2, 0)
        return im

    def pre_process_im(self, im):
        if len(im.shape) == 2 or im.shape[2] == 1:
            im = self.replicate_dim(im)
        # Resize.
        if (self.resize_h_w is not None) \
                and (self.resize_h_w != (im.shape[0], im.shape[1])):
            im = cv2.resize(im, self.resize_h_w[::-1], interpolation=cv2.INTER_LINEAR)

        if (self.crop_h_w is not None) \
            and (self.crop_h_w != (im.shape[0], im.shape[1])):
            im = self.rand_crop_im(im, self.crop_h_w)

        # scaled by 1/255.
        if self.scale:
            im = im / 255.

        # Subtract mean and scaled by std
        # im -= np.array(self.im_mean) # This causes an error:
        # Cannot cast ufunc subtract output from dtype('float64') to
        # dtype('uint8') with casting rule 'same_kind'
        if self.im_mean is not None:
            im = im - np.array(self.im_mean)
        if self.im_mean is not None and self.im_std is not None:
            im = im / np.array(self.im_std).astype(float)

        # May mirror image.
        mirrored = False
        if self.mirror and self.prng.uniform() > 0.5:
            im = im[:, ::-1, :]


        # The original image has dims 'HWC', transform it to 'CHW'.
        if self.batch_dims == 'NCHW':
            im = im.transpose(2, 0, 1)

        return im
