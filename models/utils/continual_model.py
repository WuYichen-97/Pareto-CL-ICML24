# Copyright 2020-present, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Davide Abati, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch.nn as nn
from torch.optim import SGD
import torch
import torchvision
from argparse import Namespace
from utils.conf import get_device


class ContinualModel(nn.Module):
    """
    Continual learning model.
    """
    NAME = None
    COMPATIBILITY = []

    def __init__(self, backbone: nn.Module, loss: nn.Module,
                args: Namespace, transform: torchvision.transforms) -> None:
        super(ContinualModel, self).__init__()

        self.net = backbone
        # self.net = self.net #.main
        self.loss = loss
        self.args = args
        self.transform = transform
        self.opt = SGD(self.net.parameters(), lr=self.args.lr)
        # self.opt = SGD(filter(lambda p: p.requires_grad and 'classifier1' not in p.name, self.net.parameters()), lr = self.args.lr)
        # self.opt1 = SGD(self.net.parameters(), lr=0.1)
        # self.opt1 = SGD(self.net.classifier1.parameters(), lr=self.args.lr) # obc
        # self.opt1 = SGD(self.net.classifier.parameters(), lr=self.args.lr)
        # self.opt1 = SGD(self.net.fc.parameters(), lr = self.args.lr)
        # self.opt1 = SGD(filter(lambda p: p.requires_grad and 'fc' in p.name, self.net.parameters()), lr = self.args.lr)
        self.device = get_device()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Computes a forward pass.
        :param x: batch of inputs
        :param task_label: some models require the task label
        :return: the result of the computation
        """
        return self.net(x)

    def observe(self, inputs: torch.Tensor, labels: torch.Tensor,
                not_aug_inputs: torch.Tensor) -> float:
        """
        Compute a training step over a given batch of examples.
        :param inputs: batch of examples
        :param labels: ground-truth labels
        :param kwargs: some methods could require additional parameters
        :return: the value of the loss function
        """
        pass
