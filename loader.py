from skimage.feature import local_binary_pattern
from sklearn.model_selection import train_test_split
from skimage import io
from skimage.util import img_as_float32
from skimage.transform import resize
from skimage import color
import torch
from torch.utils.data import Dataset
from sklearn.utils import class_weight
import numpy as np
from additions import rescale
from torchvision.transforms import ToTensor
import cv2

class Loader(Dataset):
    def __init__(self, image_path_dict,
                 names, labels, transform=None,
                 img_size=(224,224)):
        super(Loader, self).__init__()

        self.path = image_path_dict
        self.names = names
        self.labels = labels
        self.weighting = True
        self.transform = transform  # augmentation transforms
        self.img_size=img_size

        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        label = self.labels[index]
        img_path  = self.path[self.names[index]]
        image = cv2.imread(img_path)
        image = cv2.resize(image, self.img_size)

        if self.transform is not None:
            image = self.transform(image)
        return image, label


class LoaderSmall(Dataset):
    def __init__(self, image_path_dict, names, labels,
                 weighting=True,
                 weights=None,
                 train=True, transform=None, color_space='rgb'):
        """
        Args:

        """
        self.path = image_path_dict
        self.train = train
        self.weighting = weighting
        self.transform = transform  # augmentation transforms
        self.names = names
        self.labels = np.asarray(labels)
        '''self.train_names, self.test_names, \
        self.train_labels, self.test_labels = train_test_split(
            np.asarray(data),
            np.asarray(labels),
            test_size=0.15)'''
        self.color_transform_dict = {
            'rgb': color.rgb2rgbcie,
            'hed': color.rgb2hed,
            'hsv': color.rgb2hsv,
            'lab': color.rgb2lab,
            'lbp': self.rgb_lbp,
            None: None}
        self.weights = weights
        if self.weighting:
            self.weights = class_weight.compute_class_weight(
                'balanced',
                np.unique(self.labels),
                self.labels)
        if self.color_transform_dict[color_space] is not None:
            self.data = np.asarray([
                self.color_transform_dict[color_space](
                    io.imread(
                        self.path[name])) for name in self.names])
        else:
            self.data = np.asarray([
                io.imread(
                    self.path[name]) for name in self.names])
        self.labels = torch.from_numpy(self.labels)
        # self.train_data = (self.train_data-self.train_data.mean())/self.train_data.std()

    def __len__(self):
        return len(self.data)

    def rgb_lbp(self, image, P=9, R=1):
        result = np.zeros_like(image)
        result[..., 0] = local_binary_pattern(image[..., 0], P=P, R=R, method='uniform')
        result[..., 1] = local_binary_pattern(image[..., 1], P=P, R=R, method='uniform')
        result[..., 2] = local_binary_pattern(image[..., 2], P=P, R=R, method='uniform')
        return rescale(result)

    def __getitem__(self, index):
        if self.transform is not None:
            image = self.transform(self.data[index])
            label = self.labels[index]
        else:
            image = ToTensor()(self.data[index])
            label = self.labels[index]

        return image, label


class LoaderSmall_old(Dataset):
    def __init__(self, image_path_dict, labels, weighting=True,
                 train=True, transform=None, color_space='rgb'):
        """
        Args:

        """
        data = list(image_path_dict.keys())  # image ids
        self.path = image_path_dict
        self.train = train
        self.weighting = weighting
        self.transform = transform  # augmentation transforms
        self.train_names, self.test_names, \
        self.train_labels, self.test_labels = train_test_split(
            np.asarray(data),
            np.asarray(labels),
            test_size=0.15)
        self.color_transform_dict = {
            'rgb': color.rgb2rgbcie,
            'hed': color.rgb2hed,
            'hsv': color.rgb2hsv,
            'lab': color.rgb2lab,
            'lbp': self.rgb_lbp,
            None: None}

        if self.train:
            if self.weighting:
                self.weights = class_weight.compute_class_weight(
                    'balanced',
                    np.unique(
                        self.train_labels),
                    self.train_labels)
            if self.color_transform_dict[color_space] is not None:
                self.train_data = np.asarray([
                    self.color_transform_dict[color_space](
                        io.imread(
                            self.path[name])) for name in self.train_names])
            else:
                self.train_data = np.asarray([
                    io.imread(
                        self.path[name]) for name in self.train_names])
            self.train_labels = torch.from_numpy(self.train_labels)
            # self.train_data = (self.train_data-self.train_data.mean())/self.train_data.std()

        else:
            if self.weighting:
                self.weights = class_weight.compute_class_weight(
                    'balanced',
                    np.unique(
                        self.test_labels),
                    self.test_labels)
            if self.color_transform_dict[color_space] is not None:
                self.test_data = np.asarray([
                    self.color_transform_dict[color_space](
                        io.imread(
                            self.path[name])) for name in self.test_names])
            else:
                self.test_data = np.asarray([
                    io.imread(
                        self.path[name]) for name in self.test_names])
            self.test_labels = torch.from_numpy(self.test_labels)
            #self.test_data = (self.test_data - self.test_data.mean()) / self.test_data.std()

    def __len__(self):
        if self.train:
            return len(self.train_data)
        else:
            return len(self.test_data)

    def rgb_lbp(self, image, P=9, R=1):
        result = np.zeros_like(image)
        result[..., 0] = local_binary_pattern(image[..., 0], P=P, R=R, method='uniform')
        result[..., 1] = local_binary_pattern(image[..., 1], P=P, R=R, method='uniform')
        result[..., 2] = local_binary_pattern(image[..., 2], P=P, R=R, method='uniform')
        return rescale(result)

    def __getitem__(self, index):
        if self.train:
            if self.transform is not None:
                image = self.transform(self.train_data[index])
                    #**{'image': self.train_data[index]})['image']
                label = self.train_labels[index]
            else:
                image, label = ToTensor()(
                    self.train_data[index]), self.train_labels[index]
        else:
            if self.transform is not None:
                image = self.transform(self.test_data[index])#**{'image': self.train_data[index]})['image']
                label = self.test_labels[index]
            else:
                image, label = ToTensor()(
                self.test_data[index]), self.test_labels[index]
        return image, label


class SegmentationLoader(Dataset):
    def __init__(self, image_path_dict, mask_path_dict,
                 labels,
                 train=True,
                 transform=None,
                 color_space='rgb'):
        data = list(image_path_dict.keys())  # image ids
        self.path = image_path_dict
        self.masks = mask_path_dict
        self.train = train
        self.transform = transform  # augmentation transforms
        self.train_names, self.test_names, \
        self.train_labels, self.test_labels = train_test_split(
            np.asarray(data),
            np.asarray(labels),
            test_size=0.15)
        self.color_transform_dict = {
            'rgb': color.rgb2rgbcie,
            'hed': color.rgb2hed,
            'hsv': color.rgb2hsv,
            'lab': color.rgb2lab,
            'lbp': self.rgb_lbp,
            None: None}
