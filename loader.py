from sklearn.model_selection import train_test_split
from skimage import io
from skimage.transform import resize
from skimage import color
import torch
from torch.utils.data import Dataset as Dataset
import os
import numpy as np
from additions import rescale


class Loader(Dataset):
    def __init__(self,  path, labels, image_name=None,
                 train=True, transform=None, color_space='rgb'):
        """
        Args:
            image_name (string): Name of the file in the image
            train (bool): whether to load training set or testing set
            path (string): Path to the folder with images.
            labels (np.ndarray): labels for images.
            transform (callable, optional): Optional transform to be applied
            on a sample.
        """
        data = sorted(os.listdir(path))
        self.path = path
        self.train = train
        self.name = image_name
        self.transform = transform
        self.train_names, self.test_names, \
            self.train_labels,  self.test_labels = train_test_split(
                np.asarray(data),
                np.asarray(labels),
                test_size=0.1)
        self.color_transform_dict = {
            'rgb': color.rgb2rgbcie,
            'hed': color.rgb2hed,
            'hsv': color.rgb2hsv, None: None}

        if self.train:
            if self.color_transform_dict[color_space] is not None:
                self.train_data = torch.from_numpy(
                    np.asarray([rescale(np.transpose(
                        resize(self.color_transform_dict[color_space](
                            io.imread(
                                os.path.join(self.path, name))),
                               (224, 224),
                               mode='reflect').astype('float32'),
                        (2, 1, 0))) for name in self.train_names]))
            else:
                self.train_data = torch.from_numpy(
                    np.asarray([rescale(
                        np.transpose(resize(io.imread(os.path.join(
                            self.path, name)),
                            (224, 224),
                            mode='reflect').astype('float32'),
                            (2, 1, 0))) for name in self.train_names]))
            self.train_labels = torch.from_numpy(self.train_labels)
        else:
            if self.color_transform_dict[color_space] is not None:
                self.test_data = torch.from_numpy(
                    np.asarray([rescale(np.transpose(
                        resize(self.color_transform_dict[color_space](
                            io.imread(
                                os.path.join(self.path, name))),
                               (224, 224),
                               mode='reflect').astype('float32'),
                        (2, 1, 0))) for name in self.test_names]))
            else:
                self.test_data = torch.from_numpy(
                    np.asarray([rescale(
                        np.transpose(resize(io.imread(os.path.join(
                            self.path, name)),
                            (224, 224),
                            mode='reflect').astype('float32'),
                            (2, 1, 0))) for name in self.test_names]))
            self.test_labels = torch.from_numpy(self.test_labels)

    def __len__(self):
        if self.train:
            return len(self.train_data)
        else:
            return len(self.test_data)

    def __getitem__(self, index):
        if self.train:
            image, label = self.train_data[index], self.train_labels[index]
        else:
            image, label = self.test_data[index], self.test_labels[index]
        return image, label
