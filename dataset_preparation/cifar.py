from __future__ import print_function

import os
import os.path
from torchvision.transforms import transforms
import numpy as np
import sys
from PIL import Image
from torchvision.datasets import VisionDataset
from utils.const import *

if sys.version_info[0] == 2:
    import cPickle as pickle
else:
    import pickle


class CIFAR10(VisionDataset):
    trans = transforms.Compose([
        transforms.Resize(NF, interpolation=Image.LANCZOS),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    def __init__(self, train_list, image_size, train=True, target_transform=None):

        super(CIFAR10, self).__init__(CIFAR_ROOT, transform=self.trans,
                                      target_transform=target_transform)
        self.train = train
        self.data = []
        self.targets = []
        self.train_list = train_list

        for file_name in self.train_list:
            file_path = os.path.join(CIFAR_ROOT, BATCH_FOLDER, file_name)
            with open(file_path, 'rb') as f:
                entry = pickle.load(f, encoding='latin1')
                self.data.append(entry['data'])
                if 'labels' in entry:
                    self.targets.extend(entry['labels'])
                else:
                    self.targets.extend(entry['fine_labels'])

        self.data = np.vstack(self.data).reshape(-1, 3, image_size, image_size)
        self.data = self.data.transpose((0, 2, 3, 1))

    def __getitem__(self, index):

        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(img.astype('uint8'))
        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.data)

    def extra_repr(self):
        return "Split: {}".format("Train" if self.train is True else "Test")
