import numpy as np
from skimage.io import imshow, imread
import torchvision
from torchvision import transforms
from torchvision.utils import make_grid
from imgaug import augmenters as iaa
import imgaug as ia

import matplotlib.pyplot as plt
import matplotlib as mpl
%matplotlib inline
mpl.rcParams['axes.grid'] = False
mpl.rcParams['image.interpolation'] = 'nearest'
mpl.rcParams['figure.figsize'] = 15, 10

def show(img):
    npimg = img.numpy()
    plt.figure()
    plt.imshow(np.transpose(npimg, (1, 2, 0)), interpolation='nearest')

import os
import random

import torch
import torch.utils.data as data

from PIL import Image


class SegmentationDatasetImgaug(data.Dataset):
    IMG_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif']

    @staticmethod
    def _isimage(image, ends):
        return any(image.endswith(end) for end in ends)  # 只要是以上那些扩展名的图片就会返回True

    @staticmethod
    def _load_input_image(path):
        return imread(path)  # 花哨的imread()写法

    @staticmethod
    def _load_target_image(path):
        return imread(path, as_gray=True)[..., np.newaxis] # 也是花哨的imread()写法，虽然固定是返回灰度图。在最后一个维度加np.newaxis，等价于加一个维度

    def __init__(self, input_root, target_root, transform=None, input_only=None):
        self.input_root = input_root # input_root=图像文件目录
        self.target_root = target_root # target_root=标签文件目录
        self.transform = transform # 要进行的图像增强
        self.input_only = input_only # 这些图像增强(input_only)只对图像文件有效

        self.input_ids = sorted(
            img for img in os.listdir(self.input_root)
                if self._isimage(img, self.IMG_EXTENSIONS)
        )  # 返回一个包含图像文件目录下的所有图像，并将它们按照文件路径升序排序后的列表

        self.target_ids = sorted(
            img for img in os.listdir(self.target_root)
                if self._isimage(img, self.IMG_EXTENSIONS)
        )  # 跟上面类似

        # 如果input_ids和target_ids长度对不上就会返回AssertionError
        assert (len(self.input_ids) == len(self.target_ids))

    def _activator_masks(self, images, augmenter, parents, default):
        # 如果input_only跟要进行的图像增强同名，就返回False
        if self.input_only and augmenter.name in self.input_only:
            return False
        else:
            return default

    def __getitem__(self, idx):
        input_img = self._load_input_image(
            os.path.join(self.input_root, self.input_ids[idx]))
        target_img = self._load_target_image(
            os.path.join(self.target_root, self.target_ids[idx]))

        if self.transform:
            det_tf = self.transform.to_deterministic()
            input_img = det_tf.augment_image(input_img)
            target_img = det_tf.augment_image(
                target_img,
                hooks=ia.HooksImages(activator=self._activator_masks))

        to_tensor = transforms.ToTensor()
        input_img = to_tensor(input_img)
        target_img = to_tensor(target_img)

        return input_img, target_img, self.input_ids[idx]

    def __len__(self):
        return len(self.input_ids)


augs = iaa.Sequential([
    iaa.Scale((299, 299)),
    iaa.Fliplr(0.5),
    iaa.Affine(rotate=(-45, 45),
               translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)}),
    iaa.Add((-40, 40), per_channel=0.5, name="color-jitter")
])

ds3 = SegmentationDatasetImgaug(
    '../data/segmentation/input/', '../data/segmentation/masks/',
    transform=augs,
    input_only=['color-jitter']
)