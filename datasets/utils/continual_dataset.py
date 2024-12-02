# Copyright 2022-present, Lorenzo Bonicelli, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from abc import abstractmethod
from argparse import Namespace
from torch import nn as nn
from torchvision.transforms import transforms
from torch.utils.data import DataLoader
from typing import Tuple
from torchvision import datasets
import numpy as np
import torch.optim
from copy import deepcopy

class ContinualDataset:
    """
    Continual learning evaluation setting.
    """
    NAME = None
    SETTING = None
    N_CLASSES_PER_TASK = None
    N_TASKS = None
    TRANSFORM = None

    def __init__(self, args: Namespace) -> None:
        """
        Initializes the train and test lists of dataloaders.
        :param args: the arguments which contains the hyperparameters
        """
        self.train_loader = None
        self.test_loaders = []
        self.i = 0
        self.args = args

    @abstractmethod
    def get_data_loaders(self) -> Tuple[DataLoader, DataLoader]:
        """
        Creates and returns the training and test loaders for the current task.
        The current training loader and all test loaders are stored in self.
        :return: the current training and test loaders
        """
        pass

    @staticmethod
    @abstractmethod
    def get_backbone() -> nn.Module:
        """
        Returns the backbone to be used for to the current dataset.
        """
        pass

    @staticmethod
    @abstractmethod
    def get_transform() -> transforms:
        """
        Returns the transform to be used for to the current dataset.
        """
        pass

    @staticmethod
    @abstractmethod
    def get_loss() -> nn.functional:
        """
        Returns the loss to be used for to the current dataset.
        """
        pass

    @staticmethod
    @abstractmethod
    def get_normalization_transform() -> transforms:
        """
        Returns the transform used for normalizing the current dataset.
        """
        pass

    @staticmethod
    @abstractmethod
    def get_denormalization_transform() -> transforms:
        """
        Returns the transform used for denormalizing the current dataset.
        """
        pass

    @staticmethod
    @abstractmethod
    def get_scheduler(model, args: Namespace) -> torch.optim.lr_scheduler:
        """
        Returns the scheduler to be used for to the current dataset.
        """
        pass

    @staticmethod
    def get_epochs():
        pass

    @staticmethod
    def get_batch_size():
        pass

    @staticmethod
    def get_minibatch_size():
        pass
    
## For Rebuttal to run the upper bound
# def store_masked_loaders(train_dataset: datasets, test_dataset: datasets,
#                     setting: ContinualDataset) -> Tuple[DataLoader, DataLoader]:
#     """
#     Divides the dataset into tasks.
#     :param train_dataset: train dataset
#     :param test_dataset: test dataset
#     :param setting: continual learning setting
#     :return: train and test loaders
#     """
#     # split the task according to the targets
#     train_mask = np.logical_and(np.array(train_dataset.targets) >= 0,
#         np.array(train_dataset.targets) < 10)
#     train_dataset.data = train_dataset.data[train_mask]
#     train_dataset.targets = np.array(train_dataset.targets)[train_mask]
#     print('len_train',len(train_dataset.data))
#     train_loader = DataLoader(train_dataset,
#                                 batch_size=setting.args.batch_size, shuffle=True, num_workers=4)
#     setting.train_loader = train_loader

#     # test loaders
#     setting.test_loaders = []
#     for index in range(5):
#         test_dataset1 = deepcopy(test_dataset)
#         test_mask = np.logical_and(np.array(test_dataset1.targets) >=2*index,
#            np.array(test_dataset.targets) < 2*index + setting.N_CLASSES_PER_TASK)
#         test_dataset1.data = test_dataset1.data[test_mask]
#         test_dataset1.targets = np.array(test_dataset1.targets)[test_mask]
#         test_loader1 = DataLoader(test_dataset,
#                                 batch_size=setting.args.batch_size, shuffle=False, num_workers=4)
#         setting.test_loaders.append(test_loader1)

#     setting.i += setting.N_CLASSES_PER_TASK  # update the task split information
#     return train_loader, test_loader1


## Exchange task order
'''
def store_masked_loaders(train_dataset: datasets, test_dataset: datasets,
                    setting: ContinualDataset, tag=False) -> Tuple[DataLoader, DataLoader]:
    """
    Divides the dataset into tasks.
    :param train_dataset: train dataset
    :param test_dataset: test dataset
    :param setting: continual learning setting
    :return: train and test loaders
    """
    # split the task according to the targets
    if setting.i == 0:
        train_mask = np.logical_and(np.array(train_dataset.targets) >= 190,
            np.array(train_dataset.targets) < 200)
    else:
        train_mask = np.logical_and(np.array(train_dataset.targets) >= setting.i,
            np.array(train_dataset.targets) < setting.i + setting.N_CLASSES_PER_TASK)
    train_dataset.data = train_dataset.data[train_mask]
    train_dataset.targets = np.array(train_dataset.targets)[train_mask]
    train_loader = DataLoader(train_dataset,
                                batch_size=setting.args.batch_size, shuffle=True, num_workers=4, drop_last=False)
    setting.train_loader = train_loader

    # test loaders
    setting.test_loaders = []
    # for index in range(5):
    # N: task number in total
    N = 20 #5
    # for index in range(setting.i//setting.N_CLASSES_PER_TASK+2):
    for index in range(N):
        if index == N:
            break
        test_dataset1 = deepcopy(test_dataset)
        # print('index', 10*index + setting.N_CLASSES_PER_TASK)
        # if index == 2:
        #     test_mask = np.logical_and(np.array(test_dataset1.targets) >= 8,
        #                  np.array(test_dataset1.targets) < 10)
        #     # print('test_mask',len(test_mask))
        # else:
        test_mask = np.logical_and(np.array(test_dataset1.targets) >=setting.N_CLASSES_PER_TASK*index,
            np.array(test_dataset1.targets) < setting.N_CLASSES_PER_TASK*index + setting.N_CLASSES_PER_TASK)
            # print('test_mask_ori',len(test_mask))
        test_dataset1.data = test_dataset1.data[test_mask]
        # print('stored_mask_length', test_dataset1.data.shape)
        test_dataset1.targets = np.array(test_dataset1.targets)[test_mask]
        test_loader1 = DataLoader(test_dataset1,
                                batch_size=setting.args.batch_size, shuffle=True, num_workers=4, drop_last=False)
        setting.test_loaders.append(test_loader1)
    if not tag:
        setting.i += setting.N_CLASSES_PER_TASK  # update the task split information
    return train_loader, test_loader1
'''

## The function to test the relationship between Task similarity and BWT/FWT
'''
def store_masked_loaders(train_dataset: datasets, test_dataset: datasets,
                    setting: ContinualDataset, tag=False) -> Tuple[DataLoader, DataLoader]:
    """
    Divides the dataset into tasks.
    :param train_dataset: train dataset
    :param test_dataset: test dataset
    :param setting: continual learning setting
    :return: train and test loaders
    """
    # split the task according to the targets
    train_mask = np.logical_and(np.array(train_dataset.targets) >= setting.i,
        np.array(train_dataset.targets) < setting.i + setting.N_CLASSES_PER_TASK)
    train_dataset.data = train_dataset.data[train_mask]
    train_dataset.targets = np.array(train_dataset.targets)[train_mask]
    train_loader = DataLoader(train_dataset,
                                batch_size=setting.args.batch_size, shuffle=True, num_workers=4, drop_last=False)
    setting.train_loader = train_loader

    # test loaders
    setting.test_loaders = []
    # for index in range(5):
    # N: task number in total
    # N = 5
    # for index in range(N):
    for index in range(setting.i//setting.N_CLASSES_PER_TASK+2):
        # print('index',index)
        if index == 5:
            break
        test_dataset1 = deepcopy(test_dataset)
        # print('index', 10*index + setting.N_CLASSES_PER_TASK)
        test_mask = np.logical_and(np.array(test_dataset1.targets) >=setting.N_CLASSES_PER_TASK*index,
           np.array(test_dataset1.targets) < setting.N_CLASSES_PER_TASK*index + setting.N_CLASSES_PER_TASK)
        test_dataset1.data = test_dataset1.data[test_mask]
        # print('stored_mask_length', test_dataset1.data.shape)
        test_dataset1.targets = np.array(test_dataset1.targets)[test_mask]
        test_loader1 = DataLoader(test_dataset1,
                                batch_size=setting.args.batch_size, shuffle=True, num_workers=4, drop_last=False)
        setting.test_loaders.append(test_loader1)
    if not tag:
        setting.i += setting.N_CLASSES_PER_TASK  # update the task split information
    return train_loader, test_loader1
'''

# Pretrained ResNet18 Cifar100 order (do pretrained ... WACV wangyuxiong)
'''
# task_order = np.array([ [26, 86, 2, 55, 75],
#              [93, 16, 73, 54, 95],
#              [53, 92, 78, 13, 7],
#              [30, 22, 24, 33, 8],
#              [43, 62, 3, 71, 45],
#              [48, 6, 99, 82, 76],
#              [60, 80, 90, 68, 51],
#              [27, 18, 56, 63, 74],
#              [1, 61, 42, 41, 4],
#              [15, 17, 40, 38, 5],
#              [91, 59, 0, 34, 28],
#              [50, 11, 35, 23, 52],
#              [10, 31, 66, 57, 79],
#              [85, 32, 84, 14, 89],
#              [19, 29, 49, 97, 98],
#              [69, 20, 94, 72, 77],
#              [25, 37, 81, 46, 39],
#              [65, 58, 12, 88, 70],
#              [87, 36, 21, 83, 9],
#              [96, 67, 64, 47, 44],
#              ])

# Vehicle 1 to Vehicle 2
task_order = np.array([[0, 1 ],
             [2, 3 ],
             [4, 5 ],
             [6, 7 ],
             [8, 9 ]
             ])
# task_order = np.array([[8, 13 ],
#              [48,58 ],
#              [90, 41 ],
#              [69, 81 ],
#              [85, 89 ]
#              ])

# define order of cifar100
def store_masked_loaders(train_dataset: datasets, test_dataset: datasets,
                    setting: ContinualDataset, tag=False) -> Tuple[DataLoader, DataLoader]:
    # split the task according to the targets
    # print('type', [ i for i in train_dataset.targets])
    # print('task_order', task_order[int(setting.i/setting.N_CLASSES_PER_TASK)])
    train_mask = np.array( [ i in task_order[int(setting.i/setting.N_CLASSES_PER_TASK)] 
                 for i in train_dataset.targets  ]  )
    # print('train_mask_len', len(train_mask))


    test_mask = np.array( [ i in task_order[int(setting.i/setting.N_CLASSES_PER_TASK)] 
                 for i in test_dataset.targets  ] )

    train_dataset.data = train_dataset.data[train_mask]
    train_dataset.targets = np.array(train_dataset.targets)[train_mask]
    train_loader = DataLoader(train_dataset,
                              batch_size=setting.args.batch_size, shuffle=True, num_workers=4, drop_last=False)
    setting.train_loader = train_loader


    # test_dataset.data = test_dataset.data[test_mask][]
    # test_dataset.targets = np.array(test_dataset.targets)[test_mask]


    # test_loader = DataLoader(test_dataset,
    #                          batch_size=setting.args.batch_size, shuffle=False, num_workers=4, drop_last=False)
    # setting.test_loaders.append(test_loader)


    # setting.i += setting.N_CLASSES_PER_TASK  # update the task split information
    
    for index in range(setting.i//setting.N_CLASSES_PER_TASK+2):
        if index == 5:
            break
        test_dataset1 = deepcopy(test_dataset)
        # print('index', 10*index + setting.N_CLASSES_PER_TASK)
        # if index == 2:
        #     test_mask = np.logical_and(np.array(test_dataset1.targets) >= 8,
        #                  np.array(test_dataset1.targets) < 10)
        #     # print('test_mask',len(test_mask))
        # else:
        test_mask = np.logical_and(np.array(test_dataset1.targets) >=setting.N_CLASSES_PER_TASK*index,
            np.array(test_dataset1.targets) < setting.N_CLASSES_PER_TASK*index + setting.N_CLASSES_PER_TASK)
            # print('test_mask_ori',len(test_mask))
        test_dataset1.data = test_dataset1.data[test_mask]
        # print('stored_mask_length', test_dataset1.data.shape)
        test_dataset1.targets = np.array(test_dataset1.targets)[test_mask]
        test_loader1 = DataLoader(test_dataset1,
                                batch_size=setting.args.batch_size, shuffle=True, num_workers=4, drop_last=False)
        setting.test_loaders.append(test_loader1)
    setting.i += setting.N_CLASSES_PER_TASK
    return train_loader, test_loader1
'''

# The original store_masked_loaders
# '''
def store_masked_loaders(train_dataset: datasets, test_dataset: datasets,
                    setting: ContinualDataset, tag=False) -> Tuple[DataLoader, DataLoader]:
    """
    Divides the dataset into tasks.
    :param train_dataset: train dataset
    :param test_dataset: test dataset
    :param setting: continual learning setting
    :return: train and test loaders
    """
    # split the task according to the targets
    train_mask = np.logical_and(np.array(train_dataset.targets) >= setting.i,
        np.array(train_dataset.targets) < setting.i + setting.N_CLASSES_PER_TASK)

    test_mask = np.logical_and(np.array(test_dataset.targets) >= setting.i,
        np.array(test_dataset.targets) < setting.i + setting.N_CLASSES_PER_TASK)

    train_dataset.data = train_dataset.data[train_mask]
    test_dataset.data = test_dataset.data[test_mask]

    train_dataset.targets = np.array(train_dataset.targets)[train_mask]
    test_dataset.targets = np.array(test_dataset.targets)[test_mask]

    train_loader = DataLoader(train_dataset,
                              batch_size=setting.args.batch_size, shuffle=True, num_workers=4, drop_last=True)
    test_loader = DataLoader(test_dataset,
                             batch_size=setting.args.batch_size, shuffle=True, num_workers=4, drop_last=False)
    setting.test_loaders.append(test_loader)
    setting.train_loader = train_loader

    setting.i += setting.N_CLASSES_PER_TASK  # update the task split information
    return train_loader, test_loader
# '''

def get_previous_train_loader(train_dataset: datasets, batch_size: int,
                              setting: ContinualDataset) -> DataLoader:
    """
    Creates a dataloader for the previous task.
    :param train_dataset: the entire training set
    :param batch_size: the desired batch size
    :param setting: the continual dataset at hand
    :return: a dataloader
    """
    train_mask = np.logical_and(np.array(train_dataset.targets) >=
        setting.i - setting.N_CLASSES_PER_TASK, np.array(train_dataset.targets)
        < setting.i - setting.N_CLASSES_PER_TASK + setting.N_CLASSES_PER_TASK)

    train_dataset.data = train_dataset.data[train_mask]
    train_dataset.targets = np.array(train_dataset.targets)[train_mask]

    return DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
