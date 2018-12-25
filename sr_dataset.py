import os.path
from functools import reduce
import operator

import cv2
import numpy as np
from torch.utils.data import Dataset
import torch


def bicubic_downsample(img, downsampled_dimensions):
    return torch.nn.functional.interpolate(img.unsqueeze(0).unsqueeze(0),
                                           downsampled_dimensions,
                                           None, mode='bicubic').view(downsampled_dimensions[0], downsampled_dimensions[1])


def rotations():
    """This function should really just do transposes
    (because rotations are just flips + tranposes)
    but I'm just doing what the Matlab script did
    """

    def none(img):
        return img

    def rot_90(img):
        return img.transpose(0, 1)

    def rot_180(img):
        return img.flip(0)

    def rot_270(img):
        return img.transpose(0, 1).flip(1)

    return (none,
            rot_90,
            rot_180,
            rot_270)


def flips():
    """TODO: Figure out why tensor.flip() is so slow...
    """
    return (lambda img: img,
            lambda img: img.flip(0),
            lambda img: img.flip(1))


def scales():
    def down_dims(img, fx): return (
        int(img.shape[0] * fx), int(img.shape[1] * fx))

    return (lambda img: img,
            lambda img: bicubic_downsample(img, down_dims(img, 0.7)),
            lambda img: bicubic_downsample(img, down_dims(img, 0.5)))


def paper_transforms():
    """The transforms used the in LapSRN paper

    :returns: list of transforms to apply
    :rtype: list

    """
    def outer(scale, flip, rot):
        def inner(img):
            return scale(flip(rot(img)))

        return inner

    return [outer(scale, flip, rot) for scale in scales() for flip in flips() for rot in rotations()]


class ImageCache:
    """
    Simple class for LRU caching images with a configurable max size.
    """

    def __init__(self, getter, max_cache_size=1000000000, verbose=False):
        self.image_cache = {}
        self.getter = getter
        self.cache_size = 0
        self.max_cache_size = max_cache_size
        self.insertion_order = []
        self.verbose = verbose

    def __calc_size__(self, tensor):
        return reduce(operator.mul,
                      (dim for dim in tensor.shape), 4)

    def __getitem__(self, index):
        if index in self.image_cache.keys():
            return self.image_cache[index]

        value = self.getter(index)

        self.image_cache[index] = value
        self.insertion_order.append(index)
        self.cache_size += self.__calc_size__(value)
        if self.verbose:
            print("Inserted {} into cache; cache size is now {}".format(
                index, self.cache_size))

        if self.cache_size > self.max_cache_size:
            remove_index = self.insertion_order.pop(0)
            remove_size = self.__calc_size__(self.image_cache[remove_index])
            del self.image_cache[remove_index]
            if self.verbose:
                print("Removed {} from cache".format(remove_index))
            self.cache_size -= max(remove_size, 0)

        return value


class SRImageDataset(Dataset):
    """
    Dataset for super-resolution tasks. Accepts high-resolution image input
    from a folder, performs image augmentation, and provides an iterator
    interface for high-resolution and low-resolution patches.
    """

    def __init__(self, hr_path, patch_resolution, patch_stride,
                 cpu_tensors=False,
                 lr_transform=bicubic_downsample,
                 augment=paper_transforms(),
                 image_cache_size=60000000,
                 verbose=True):
        """Initializes super-resolution image dataset

        :param hr_path: path to HR images
        :param patch_resolution: resolution of patches to use for training
        :param patch_stride: stride for gathering patches in HR image space
        :param cpu_tensors: enable to use CPU tensors rather than CUDA tensors (allows
                        multiple threads)
        :param lr_transform: transform to convert tensors from HR to LR space. Takes
                             a function that accepts (image, LR dimensions) as input
        :param augment: a list of transforms to apply to each image for augmentation.
                        See ``paper_transforms`` for an example
        :param image_cache_size: base size of all image caches
        :param verbose: verbose output
        """
        self.mapped_images = []
        self.lr_transform = lr_transform
        self.patch_resolution = patch_resolution
        self.patch_stride = patch_stride
        self.augment = augment
        self.num_augmentations = len(augment)
        self.image_cache = ImageCache(
            self.__load_image__, image_cache_size * 10)
        self.augment_image_cache = ImageCache(
            self.__load_augment_image__, image_cache_size)
        self.cpu_tensors = cpu_tensors
        self.verbose = verbose

        def load_augment_transform(factor):
            def inner(index):
                aug_img = self.augment_image_cache[index]
                return lr_transform(
                    aug_img, (aug_img.shape[0] // factor,
                              aug_img.shape[1] // factor))
            return inner

        self.lr_x2_cache = ImageCache(
            load_augment_transform(2), image_cache_size)
        self.lr_x1_cache = ImageCache(
            load_augment_transform(4), image_cache_size)

        # patch index -> (augment image index, region in image)
        self.patch_region_lookup = {}

        if self.verbose:
            print("Caching images in memory...")

        patch_index = 0
        for image_index, file_name in enumerate(os.listdir(hr_path)):
            if self.verbose:
                print("Loaded image {}".format(image_index))
            file_path = os.path.join(hr_path, file_name)
            with open(file_path, 'rb') as fd:
                self.mapped_images.append(fd.read())
            for augment_index in range(image_index * self.num_augmentations, (image_index + 1) * self.num_augmentations):
                img = self.augment_image_cache[augment_index]

                x_partitions = (img.shape[0] // patch_stride) - 1
                y_partitions = (img.shape[1] // patch_stride) - 1
                num_patches = x_partitions * y_partitions
                for patch in range(num_patches):
                    x = (patch % x_partitions) * patch_stride
                    y = (patch // x_partitions) * patch_stride
                    self.patch_region_lookup[patch +
                                             patch_index] = (augment_index, (x, y))
                patch_index += num_patches

        if self.verbose:
            print("Done!")

    def __load_image__(self, index):
        buf = np.frombuffer(self.mapped_images[index], np.uint8)
        tensor = torch.from_numpy(cv2.cvtColor(cv2.imdecode(
            buf, cv2.IMREAD_UNCHANGED), cv2.COLOR_BGR2YUV)[:, :, 0].astype(np.float32) / 255.0)
        if not self.cpu_tensors:
            tensor = tensor.cuda()
        return tensor

    def __load_augment_image__(self, augment_index, use_tensor=True):
        img = self.image_cache[augment_index //
                               self.num_augmentations]
        tensor = self.augment[augment_index % self.num_augmentations](img)
        return tensor

    def __getitem__(self, patch_index):
        index, loc = self.patch_region_lookup[patch_index]

        hr = self.augment_image_cache[index]
        x2 = self.lr_x2_cache[index]
        inp = self.lr_x1_cache[index]

        x1_hr, x2_hr = loc[0], loc[0] + self.patch_resolution
        y1_hr, y2_hr = loc[1], loc[1] + self.patch_resolution

        x1_2, x2_2 = loc[0] // 2, (loc[0] + self.patch_resolution) // 2
        y1_2, y2_2 = loc[1] // 2, (loc[1] + self.patch_resolution) // 2

        x1_inp, x2_inp = loc[0] // 4, (loc[0] + self.patch_resolution) // 4
        y1_inp, y2_inp = loc[1] // 4, (loc[1] + self.patch_resolution) // 4

        patch_hr = hr[x1_hr:x2_hr, y1_hr:y2_hr]
        patch_2 = x2[x1_2:x2_2, y1_2:y2_2]
        patch_inp = inp[x1_inp:x2_inp, y1_inp:y2_inp]

        if self.verbose:
            assert patch_hr.shape == (
                self.patch_resolution, self.patch_resolution)
            assert patch_2.shape == (
                self.patch_resolution // 2, self.patch_resolution // 2)
            assert patch_inp.shape == (
                self.patch_resolution // 4, self.patch_resolution // 4)

        return (patch_inp.unsqueeze(0),
                patch_2.unsqueeze(0),
                patch_hr.unsqueeze(0))

    def __len__(self):
        return len(self.patch_region_lookup.keys())
