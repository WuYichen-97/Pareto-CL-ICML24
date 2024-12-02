# Copyright 2022-present, Lorenzo Bonicelli, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
'''
from torchvision.datasets import CIFAR100
import torchvision.transforms as transforms
# from backbone.ResNet18_MAML import resnet18_maml
# from backbone.ResNet18 import resnet18
from backbone.ResNet18_obc import resnet18
# from backbone.Pretrained_ResNet18 import ResNet18
from backbone.pc_cnn import PC_CNN
import torch.nn.functional as F
from utils.conf import base_path
from PIL import Image
from datasets.utils.validation import get_train_val
from datasets.utils.continual_dataset import ContinualDataset, store_masked_loaders
from typing import Tuple
from datasets.transforms.denormalization import DeNormalize
import torch.optim

import torchvision.models as models
import torch.nn as nn
import PIL

# ResNet18 = models.resnet18(pretrained=False)
# num_ftrs = ResNet18.fc.in_features #512
# ResNet18.fc = nn.Linear(num_ftrs,100)
# # 将模型的第一个卷积层的kernel size从7改为3
# new_conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
# # 将模型的第一个卷积层的权重复制到新的卷积层中
# new_conv1.weight.data[:, :, :3, :3] = ResNet18.conv1.weight.data[:, :, :3, :3]
# # 将新的卷积层替换模型的第一个卷积层
# ResNet18.conv1 = new_conv1
# pretrained_dict = ResNet18.state_dict()

class TCIFAR100(CIFAR100):
    def __init__(self, root, train=True, transform=None,
                 target_transform=None, download=False) -> None:
        self.root = root
        super(TCIFAR100, self).__init__(root, train, transform, target_transform, download=not self._check_integrity())

class MyCIFAR100(CIFAR100):
    """
    Overrides the CIFAR100 dataset to change the getitem function.
    """
    def __init__(self, root, train=True, transform=None,
                 target_transform=None, download=False) -> None:
        self.not_aug_transform = transforms.Compose([transforms.ToTensor()])
        self.root = root
        super(MyCIFAR100, self).__init__(root, train, transform, target_transform, not self._check_integrity())

    def __getitem__(self, index: int) -> Tuple[type(Image), int, type(Image)]:
        """
        Gets the requested element from the dataset.
        :param index: index of the element to be returned
        :returns: tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]

        # to return a PIL Image
        img = Image.fromarray(img, mode='RGB')
        original_img = img.copy()

        not_aug_img = self.not_aug_transform(original_img)
        # print('not_aug', not_aug_img.shape)
        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        if hasattr(self, 'logits'):
            return img, target, not_aug_img, self.logits[index]
        # print('img', img)

        return img, target, not_aug_img


class SequentialCIFAR100(ContinualDataset):

    NAME = 'seq-cifar100'
    SETTING = 'class-il'
    N_CLASSES_PER_TASK = 5 #5
    N_TASKS = 20 #20
    TRANSFORM = transforms.Compose(
            [transforms.RandomCrop(32, padding=4),
             transforms.RandomHorizontalFlip(),
             transforms.ToTensor(),
             transforms.Normalize((0.5071, 0.4867, 0.4408),
                                  (0.2675, 0.2565, 0.2761))])
    # mean, std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
    # TRANSFORM = transforms.Compose([
    #         transforms.Resize(224, interpolation=PIL.Image.BICUBIC),
    #         transforms.ToTensor(),
    #         transforms.Normalize(mean, std)
    #         ])

    def get_examples_number(self):
        train_dataset = MyCIFAR100(base_path() + 'CIFAR10', train=True,
                                  download=True)
        return len(train_dataset.data)

    def get_data_loaders(self):
        transform = self.TRANSFORM

        test_transform = transforms.Compose(
            [transforms.ToTensor(), self.get_normalization_transform()])

        # mean, std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
        # train_transform = transforms.Compose([
        #     transforms.Resize(224, interpolation=PIL.Image.BICUBIC),
        #     transforms.ToTensor(),
        #     transforms.Normalize(mean, std)
        #     ])


        train_dataset = MyCIFAR100(base_path() + 'CIFAR100', train=True,
                                  download=True, transform=transform)
                                # download=True, transform=train_transform)
        if self.args.validation:
            train_dataset, test_dataset = get_train_val(train_dataset,
                                                    test_transform, self.NAME)
        else:
            test_dataset = TCIFAR100(base_path() + 'CIFAR100',train=False,
                                   download=True, transform=test_transform)
                                # download=True, transform=transform)

        train, test = store_masked_loaders(train_dataset, test_dataset, self)
        
        return train, test

    @staticmethod
    def get_transform():
        transform = transforms.Compose(
            [transforms.ToPILImage(), SequentialCIFAR100.TRANSFORM])
        return transform

    @staticmethod
    # def get_backbone():
    #     return ResNet18(100)
    def get_backbone():
        return resnet18(SequentialCIFAR100.N_CLASSES_PER_TASK
                        * SequentialCIFAR100.N_TASKS)
    # def get_backbone():
    #     return resnet18_maml(SequentialCIFAR100.N_CLASSES_PER_TASK
    #                     * SequentialCIFAR100.N_TASKS)
    # def get_backbone():
    #     return PC_CNN(3*32*32,SequentialCIFAR100.N_CLASSES_PER_TASK * SequentialCIFAR100.N_TASKS)

    @staticmethod
    def get_loss():
        return F.cross_entropy

    @staticmethod
    def get_normalization_transform():
        transform = transforms.Normalize((0.5071, 0.4867, 0.4408),
                                         (0.2675, 0.2565, 0.2761))
        return transform

    @staticmethod
    def get_denormalization_transform():
        transform = DeNormalize((0.5071, 0.4867, 0.4408),
                                (0.2675, 0.2565, 0.2761))
        return transform

    @staticmethod
    def get_epochs():
        return 50

    @staticmethod
    def get_batch_size():
        return 32

    @staticmethod
    def get_minibatch_size():
        return SequentialCIFAR100.get_batch_size()

    @staticmethod
    def get_scheduler(model, args) -> torch.optim.lr_scheduler:
        model.opt = torch.optim.SGD(model.net.parameters(), lr=args.lr, weight_decay=args.optim_wd, momentum=args.optim_mom)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(model.opt, [35, 45], gamma=0.1)#, verbose=False)
        return scheduler
'''




from torchvision.datasets import CIFAR100
import torchvision.transforms as transforms
from backbone.ResNet18_MAML1 import resnet18_maml
# from backbone.ResNet18 import resnet18
from backbone.ResNet18_OBC import resnet18
# from backbone.Pretrained_ResNet18 import ResNet18

from backbone.pc_cnn import PC_CNN
import torch.nn.functional as F
from utils.conf import base_path
from PIL import Image
from datasets.utils.validation import get_train_val
from datasets.utils.continual_dataset import ContinualDataset, store_masked_loaders
from typing import Tuple
from datasets.transforms.denormalization import DeNormalize
import torch.optim

import torchvision.models as models
import torch.nn as nn
import PIL

# ResNet18 = models.resnet18(pretrained=False)
# num_ftrs = ResNet18.fc.in_features #512
# ResNet18.fc = nn.Linear(num_ftrs,100)
# # # 将模型的第一个卷积层的kernel size从7改为3
# # new_conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
# # # 将模型的第一个卷积层的权重复制到新的卷积层中
# # new_conv1.weight.data[:, :, :3, :3] = ResNet18.conv1.weight.data[:, :, :3, :3]
# # # 将新的卷积层替换模型的第一个卷积层
# # ResNet18.conv1 = new_conv1
# pretrained_dict = ResNet18.state_dict()

class TCIFAR100(CIFAR100):
    def __init__(self, root, train=True, transform=None,
                 target_transform=None, download=False) -> None:
        self.root = root
        super(TCIFAR100, self).__init__(root, train, transform, target_transform, download=not self._check_integrity())

class MyCIFAR100(CIFAR100):
    """
    Overrides the CIFAR100 dataset to change the getitem function.
    """
    def __init__(self, root, train=True, transform=None,
                 target_transform=None, download=False) -> None:
        self.not_aug_transform = transforms.Compose([transforms.ToTensor()])
        self.root = root
        super(MyCIFAR100, self).__init__(root, train, transform, target_transform, not self._check_integrity())

    def __getitem__(self, index: int) -> Tuple[type(Image), int, type(Image)]:
        """
        Gets the requested element from the dataset.
        :param index: index of the element to be returned
        :returns: tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]

        # to return a PIL Image
        img = Image.fromarray(img, mode='RGB')
        original_img = img.copy()

        not_aug_img = self.not_aug_transform(original_img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        if hasattr(self, 'logits'):
            return img, target, not_aug_img, self.logits[index]

        return img, target, not_aug_img


class SequentialCIFAR100(ContinualDataset):

    NAME = 'seq-cifar100'
    SETTING = 'class-il'
    N_CLASSES_PER_TASK = 20 #5  2
    N_TASKS = 5 #20 50
    task_id = 0
    TRANSFORM = transforms.Compose(
            [transforms.RandomCrop(32, padding=4),
             transforms.RandomHorizontalFlip(),
             transforms.ToTensor(),
             transforms.Normalize((0.5071, 0.4867, 0.4408),
                                  (0.2675, 0.2565, 0.2761))])

    # mean, std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
    # TRANSFORM = transforms.Compose([
    #         transforms.Resize(224, interpolation=PIL.Image.BICUBIC),
    #         transforms.ToTensor(),
    #         transforms.Normalize(mean, std)
    #         ])

    def get_examples_number(self):
        train_dataset = MyCIFAR100(base_path() + 'CIFAR10', train=True,
                                  download=True)
        return len(train_dataset.data)

    def get_data_loaders(self, tag=False):
        transform = self.TRANSFORM

        test_transform = transforms.Compose(
            [transforms.ToTensor(), self.get_normalization_transform()])

        mean, std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
        # train_transform = transforms.Compose([
        #     transforms.Resize(224, interpolation=PIL.Image.BICUBIC),
        #     transforms.ToTensor(),
        #     transforms.Normalize(mean, std)
        #     ])

        train_dataset = MyCIFAR100(base_path() + 'CIFAR100', train=True,
                                    download=True, transform=transform)
                                # download=True, transform=train_transform)
                                #   download=True, transform=test_transform)
        # change the label  V1 to V2

        # for i in range(len(train_dataset)):
        #     if train_dataset.targets[i] == 8:
        #         train_dataset.targets[i] = 0
        #     elif train_dataset.targets[i] == 13:
        #         train_dataset.targets[i] = 1
        #     elif train_dataset.targets[i] == 48:
        #         train_dataset.targets[i] = 2
        #     elif train_dataset.targets[i] == 58:
        #         train_dataset.targets[i] = 3
        #     elif train_dataset.targets[i] == 90:
        #         train_dataset.targets[i] = 4
        #     elif train_dataset.targets[i] == 41:
        #         train_dataset.targets[i] = 5   
        #     elif train_dataset.targets[i] == 69:
        #         train_dataset.targets[i] = 6 
        #     elif train_dataset.targets[i] == 81:
        #         train_dataset.targets[i] = 7  
        #     elif train_dataset.targets[i] == 85:
        #         train_dataset.targets[i] = 8 
        #     elif train_dataset.targets[i] == 89:
        #         train_dataset.targets[i] = 9              

        #     elif train_dataset.targets[i] == 0:
        #         train_dataset.targets[i] = 8
        #     elif train_dataset.targets[i] == 1:
        #         train_dataset.targets[i] = 13
        #     elif train_dataset.targets[i] == 2:
        #         train_dataset.targets[i] = 48
        #     elif train_dataset.targets[i] == 3:
        #         train_dataset.targets[i] = 58
        #     elif train_dataset.targets[i] == 4:
        #         train_dataset.targets[i] = 90
        #     elif train_dataset.targets[i] == 5:
        #         train_dataset.targets[i] = 41
        #     elif train_dataset.targets[i] == 6:
        #         train_dataset.targets[i] = 69
        #     elif train_dataset.targets[i] == 7:
        #         train_dataset.targets[i] = 81
        #     elif train_dataset.targets[i] == 8:
        #         train_dataset.targets[i] = 85
        #     elif train_dataset.targets[i] == 9:
        #         train_dataset.targets[i] = 89
        # '''

        # V1--> Small Mammonth

        # for i in range(len(train_dataset)):
        #     if train_dataset.targets[i] == 8:
        #         train_dataset.targets[i] = 0
        #     elif train_dataset.targets[i] == 13:
        #         train_dataset.targets[i] = 1
        #     elif train_dataset.targets[i] == 48:
        #         train_dataset.targets[i] = 2
        #     elif train_dataset.targets[i] == 58:
        #         train_dataset.targets[i] = 3
        #     elif train_dataset.targets[i] == 90:
        #         train_dataset.targets[i] = 4
        #     elif train_dataset.targets[i] == 36:
        #         train_dataset.targets[i] = 5   
        #     elif train_dataset.targets[i] == 50:
        #         train_dataset.targets[i] = 6 
        #     elif train_dataset.targets[i] == 65:
        #         train_dataset.targets[i] = 7  
        #     elif train_dataset.targets[i] == 74:
        #         train_dataset.targets[i] = 8 
        #     elif train_dataset.targets[i] == 80:
        #         train_dataset.targets[i] = 9              

        #     elif train_dataset.targets[i] == 0:
        #         train_dataset.targets[i] = 8
        #     elif train_dataset.targets[i] == 1:
        #         train_dataset.targets[i] = 13
        #     elif train_dataset.targets[i] == 2:
        #         train_dataset.targets[i] = 48
        #     elif train_dataset.targets[i] == 3:
        #         train_dataset.targets[i] = 58
        #     elif train_dataset.targets[i] == 4:
        #         train_dataset.targets[i] = 90
        #     elif train_dataset.targets[i] == 5:
        #         train_dataset.targets[i] = 36
        #     elif train_dataset.targets[i] == 6:
        #         train_dataset.targets[i] = 50
        #     elif train_dataset.targets[i] == 7:
        #         train_dataset.targets[i] = 65
        #     elif train_dataset.targets[i] == 8:
        #         train_dataset.targets[i] = 74
        #     elif train_dataset.targets[i] == 9:
        #         train_dataset.targets[i] = 80

        if self.args.validation:
            train_dataset, test_dataset = get_train_val(train_dataset,
                                                    train_transform, self.NAME)
        else:
            test_dataset = TCIFAR100(base_path() + 'CIFAR100',train=False,
                                    #   download=True, transform=train_transform)
                                   download=True, transform=test_transform)

            # for i in range(len(test_dataset)):
            #     if test_dataset.targets[i] == 8:
            #         test_dataset.targets[i] = 0
            #     elif test_dataset.targets[i] == 13:
            #         test_dataset.targets[i] = 1
            #     elif test_dataset.targets[i] == 48:
            #         test_dataset.targets[i] = 2
            #     elif test_dataset.targets[i] == 58:
            #         test_dataset.targets[i] = 3
            #     elif test_dataset.targets[i] == 90:
            #         test_dataset.targets[i] = 4
            #     elif test_dataset.targets[i] == 41:
            #         test_dataset.targets[i] = 5   
            #     elif test_dataset.targets[i] == 69:
            #         test_dataset.targets[i] = 6 
            #     elif test_dataset.targets[i] == 81:
            #         test_dataset.targets[i] = 7  
            #     elif test_dataset.targets[i] == 85:
            #         test_dataset.targets[i] = 8 
            #     elif test_dataset.targets[i] == 89:
            #         test_dataset.targets[i] = 9              

            #     elif test_dataset.targets[i] == 0:
            #         test_dataset.targets[i] = 8
            #     elif test_dataset.targets[i] == 1:
            #         test_dataset.targets[i] = 13
            #     elif test_dataset.targets[i] == 2:
            #         test_dataset.targets[i] = 48
            #     elif test_dataset.targets[i] == 3:
            #         test_dataset.targets[i] = 58
            #     elif test_dataset.targets[i] == 4:
            #         test_dataset.targets[i] = 90
            #     elif test_dataset.targets[i] == 5:
            #         test_dataset.targets[i] = 41
            #     elif test_dataset.targets[i] == 6:
            #         test_dataset.targets[i] = 69
            #     elif test_dataset.targets[i] == 7:
            #         test_dataset.targets[i] = 81
            #     elif test_dataset.targets[i] == 8:
            #         test_dataset.targets[i] = 85
            #     elif test_dataset.targets[i] == 9:
            #         test_dataset.targets[i] = 89
            
            # V1 --> Small Mammonth
 
            # for i in range(len(test_dataset)):
            #     if test_dataset.targets[i] == 8:
            #         test_dataset.targets[i] = 0
            #     elif test_dataset.targets[i] == 13:
            #         test_dataset.targets[i] = 1
            #     elif test_dataset.targets[i] == 48:
            #         test_dataset.targets[i] = 2
            #     elif test_dataset.targets[i] == 58:
            #         test_dataset.targets[i] = 3
            #     elif test_dataset.targets[i] == 90:
            #         test_dataset.targets[i] = 4
            #     elif test_dataset.targets[i] == 36:
            #         test_dataset.targets[i] = 5   
            #     elif test_dataset.targets[i] == 50:
            #         test_dataset.targets[i] = 6 
            #     elif test_dataset.targets[i] == 65:
            #         test_dataset.targets[i] = 7  
            #     elif test_dataset.targets[i] == 74:
            #         test_dataset.targets[i] = 8 
            #     elif test_dataset.targets[i] == 80:
            #         test_dataset.targets[i] = 9              

            #     elif test_dataset.targets[i] == 0:
            #         test_dataset.targets[i] = 8
            #     elif test_dataset.targets[i] == 1:
            #         test_dataset.targets[i] = 13
            #     elif test_dataset.targets[i] == 2:
            #         test_dataset.targets[i] = 48
            #     elif test_dataset.targets[i] == 3:
            #         test_dataset.targets[i] = 58
            #     elif test_dataset.targets[i] == 4:
            #         test_dataset.targets[i] = 90
            #     elif test_dataset.targets[i] == 5:
            #         test_dataset.targets[i] = 36
            #     elif test_dataset.targets[i] == 6:
            #         test_dataset.targets[i] = 50
            #     elif test_dataset.targets[i] == 7:
            #         test_dataset.targets[i] = 65
            #     elif test_dataset.targets[i] == 8:
            #         test_dataset.targets[i] = 74
            #     elif test_dataset.targets[i] == 9:
            #         test_dataset.targets[i] = 80
             

        train, test = store_masked_loaders(train_dataset, test_dataset, self, tag)
        
        return train, test

    @staticmethod
    def get_transform():
        transform = transforms.Compose(
            [transforms.ToPILImage(), SequentialCIFAR100.TRANSFORM])
        return transform

    @staticmethod
    # def get_backbone():
    #     model = resnet18(SequentialCIFAR100.N_CLASSES_PER_TASK
    #             * SequentialCIFAR100.N_TASKS)
        # my_dict = model.state_dict()
        # for name, param in pretrained_dict.items():
        #     if name in my_dict:
        #         my_dict[name].copy_(param)                        
        # # model.load_state_dict(torch.load('/apdcephfs/private_coltonwu/Continual-Learning/Pretrainned-Res18/resnet18-5c106cde.pth'))
        # return model
        # return ResNet18(100)
    # def get_backbone():
    #     return resnet18(SequentialCIFAR100.N_CLASSES_PER_TASK
    #                     * SequentialCIFAR100.N_TASKS)
    def get_backbone():
        return resnet18_maml(SequentialCIFAR100.N_CLASSES_PER_TASK
                        * SequentialCIFAR100.N_TASKS)
    # def get_backbone():
    #     return PC_CNN(3*32*32,SequentialCIFAR100.N_CLASSES_PER_TASK * SequentialCIFAR100.N_TASKS)

    @staticmethod
    def get_loss():
        return F.cross_entropy

    @staticmethod
    def get_normalization_transform():
        transform = transforms.Normalize((0.5071, 0.4867, 0.4408),
                                         (0.2675, 0.2565, 0.2761))
        return transform

    @staticmethod
    def get_denormalization_transform():
        transform = DeNormalize((0.5071, 0.4867, 0.4408),
                                (0.2675, 0.2565, 0.2761))
        return transform

    @staticmethod
    def get_epochs():
        return 50

    @staticmethod
    def get_batch_size():
        return 32

    @staticmethod
    def get_minibatch_size():
        return SequentialCIFAR100.get_batch_size()

    @staticmethod
    def get_scheduler(model, args) -> torch.optim.lr_scheduler:
        model.opt = torch.optim.SGD(model.net.parameters(), lr=args.lr, weight_decay=args.optim_wd, momentum=args.optim_mom)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(model.opt, [35, 45], gamma=0.1)#, verbose=False)
        return scheduler

