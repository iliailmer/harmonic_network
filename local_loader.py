from sklearn.model_selection import train_test_split
from skimage import io
from skimage.transform import resize
from skimage import color
import torch
from torch.utils.data import Dataset as Dataset
from sklearn.utils import class_weight
import numpy as np
from additions import rescale
from torchvision.transforms import ToTensor


class LocalLoader(Dataset):
    def __init__(self,  image_path_dict, labels, image_name=None,
                 train=True, transform=None, color_space='rgb'):
        """
        Args:

        """
        data = list(image_path_dict.keys())  # image ids
        self.path = image_path_dict
        self.train = train
        # self.name = image_name
        self.transform = transform  # augmentation transforms
        self.train_names, self.test_names, \
            self.train_labels,  self.test_labels = train_test_split(
                np.asarray(data),
                np.asarray(
                    labels),
                test_size=0.2)
        self.color_transform_dict = {
            'rgb': color.rgb2rgbcie,
            'hed': color.rgb2hed,
            'hsv': color.rgb2hsv, None: None}

        if self.train:
            self.weights = class_weight.compute_class_weight(
                'balanced',
                np.unique(
                    self.train_labels),
                self.train_labels)
            self.train_labels = torch.from_numpy(self.train_labels)
        else:
            self.weights = class_weight.compute_class_weight(
                'balanced',
                np.unique(
                    self.test_labels),
                self.test_labels)
            self.test_labels = torch.from_numpy(self.test_labels)

    def __len__(self):
        if self.train:
            return len(self.train_labels)
        else:
            return len(self.test_labels)

    def __getitem__(self, index):
        if self.train:
            label = self.train_labels[index]
            if self.transform is not None:
                if self.color_transform_dict[color_space] is not None:
                    d = {'image': rescale(
                        resize(
                            self.color_transform_dict[color_space](
                                rescale(io.imread(self.path[self.train_names[index]]).astype('float32'))),
                            (64, 64),  # (150,150)
                            mode='reflect'))}
                    image = ToTensor()(self.transform(d['image']))
                else:
                    d = {'image': rescale(
                        resize(
                            rescale(
                                io.imread(self.path[self.train_names[index]]).astype('float32')),
                            (64, 64),  # (150,150)
                            mode='reflect'))}
                    image = ToTensor()(self.transform(d['image']))
            else:
                if self.color_transform_dict[color_space] is not None:
                    d = {'image': rescale(
                        resize(
                            self.color_transform_dict[color_space](
                                rescale(io.imread(self.path[self.train_names[index]]).astype('float32'))),
                            (64, 64),  # (150,150)
                            mode='reflect'))}
                    image = ToTensor()(d['image'])
                else:
                    d = {'image': rescale(
                        resize(
                            rescale(
                                io.imread(self.path[self.train_names[index]]).astype('float32')),
                            (64, 64),  # (150,150)
                            mode='reflect'))}
                    image = ToTensor()(d['image'])
        else:
            if self.color_transform_dict[color_space] is not None:
                image = rescale(
                    resize(
                        self.color_transform_dict[color_space](
                            rescale(
                                io.imread(
                                    self.path[self.test_names[index]]).astype('float32'))),
                        (64, 64),
                        mode='reflect'))
            else:
                image = rescale(
                    resize(
                        rescale(
                            io.imread(self.path[self.test_names[index]]).astype('float32')),
                        (64, 64),
                        mode='reflect'))
            image, label = ToTensor()(
                self.test_data[index]), self.test_labels[index]
        return image, label
