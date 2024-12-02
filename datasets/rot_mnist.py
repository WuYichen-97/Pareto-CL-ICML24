# Copyright 2022-present, Lorenzo Bonicelli, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torchvision.transforms as transforms
from datasets.transforms.rotation import Rotation
from torch.utils.data import DataLoader
from backbone.MNISTMLP import MNISTMLP
from backbone.ResNet18 import resnet18
import torch.nn.functional as F
from datasets.perm_mnist import store_mnist_loaders
from datasets.utils.continual_dataset import ContinualDataset



class RotatedMNIST(ContinualDataset):
    NAME = 'rot-mnist'
    SETTING = 'domain-il'
    N_CLASSES_PER_TASK = 10
    N_TASKS = 10

    def get_data_loaders(self):
        # transform = transforms.Compose((Rotation(), transforms.ToTensor()))
        train, test = store_mnist_loaders(self)
        return train, test

    @staticmethod
    def get_backbone():
        return MNISTMLP(28 * 28, RotatedMNIST.N_CLASSES_PER_TASK)

    @staticmethod
    def get_transform():
        return None

    @staticmethod
    def get_normalization_transform():
        return None

    @staticmethod
    def get_loss():
        return F.cross_entropy

    @staticmethod
    def get_denormalization_transform():
        return None

    @staticmethod
    def get_scheduler(model, args):
        return None