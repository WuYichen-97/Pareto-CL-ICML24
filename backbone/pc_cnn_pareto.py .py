# Copyright 2022-present, Lorenzo Bonicelli, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
from backbone import MammothBackbone, xavier, num_flat_features
from collections import OrderedDict
import torch.nn.functional as F


def functional_conv_block(x, weights, bias, is_training, stride=2, padding=1):
        x = F.conv2d(x, weights, bias=bias, padding=padding,stride=stride)
        # x = F.batch_norm(x, running_mean=None, running_var=None, weight=bn_weights, bias=bn_biases,
        #                  training=is_training)
        return x

class PC_CNN(MammothBackbone):
    def __init__(self, input_size: int, output_size:int)-> None:
        super(PC_CNN, self).__init__()
        self.input_size = input_size
        self.output_size = output_size

        self.channels = 160
        self.conv1 = nn.Conv2d(3, self.channels, kernel_size = 3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(self.channels, self.channels, 3, 2, 1)
        self.conv3 = nn.Conv2d(self.channels, self.channels, 3, 2, 1)
        self.linear1 = nn.Linear(16*self.channels,320)
        self.linear2 = nn.Linear(320,320)
        self._features = nn.Sequential(
             self.conv1,
             nn.ReLU(),
             self.conv2,
             nn.ReLU(),
             self.conv3,
             nn.ReLU(),
             nn.Flatten(),
             self.linear1,
             nn.ReLU(),
             self.linear2,
             nn.ReLU(),
        )
        self.classifier = nn.Linear(320, self.output_size)
        self.net = nn.Sequential(self._features, self.classifier)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """
        Calls the Xavier parameter initialization function.
        """
        self.net.apply(xavier)


    def forward(self, x: torch.Tensor, returnt='out') -> torch.Tensor:
        """
        Compute a forward pass.
        :param x: input tensor (batch_size, input_size)
        :return: output tensor (output_size)
        """
        # x = x.view(-1, num_flat_features(x))

        feats = self._features(x)

        if returnt == 'features':
            return feats

        out = self.classifier(feats)

        if returnt == 'out':
            return out
        elif returnt == 'all':
            return (out, feats)
        raise NotImplementedError("Unknown return type")

    def functional_forward(self,x:torch.Tensor,fast_weight:OrderedDict,returnt:str ='out')-> torch.Tensor:
        x = F.relu(functional_conv_block(x, weights=fast_weight['conv1.weight'], bias = fast_weight['conv1.bias'], is_training=True))
        x = F.relu(functional_conv_block(x, weights=fast_weight['conv2.weight'], bias = fast_weight['conv2.bias'],is_training=True))
        x = F.relu(functional_conv_block(x, weights=fast_weight['conv3.weight'], bias = fast_weight['conv3.bias'],is_training=True))
        x = x.view(x.size(0), -1)
        x = F.relu(F.linear(x, fast_weight['linear1.weight'], fast_weight['linear1.bias']))
        feats = F.relu(F.linear(x, fast_weight['linear2.weight'], fast_weight['linear2.bias']))
        if returnt == 'features':
            return feats
        out = F.linear(feats, fast_weight['classifier.weight'], fast_weight['classifier.bias'])
        if returnt == 'out':
            return out
        elif returnt == 'all':
            return (out, feats)
        raise NotImplementedError("Unknown return type")

    def get_fast_weight(self) -> OrderedDict:
        return OrderedDict([[p[0], p[1].clone()] for p in self.named_parameters()])




