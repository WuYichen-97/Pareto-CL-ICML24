# Copyright 2020-present, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Davide Abati, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from utils.buffer import Buffer
from torch.nn import functional as F
from models.utils.continual_model import ContinualModel
from utils.args import *
import torch
import numpy as np
np.random.seed(0)
import torch.nn as nn
import os


def get_parser() -> ArgumentParser:
    parser = ArgumentParser(description='Continual learning via'
                                        ' Dark Experience Replay++.')
    add_management_args(parser)
    add_experiment_args(parser)
    add_rehearsal_args(parser)
    parser.add_argument('--alpha', type=float, required=True,
                        help='Penalty weight.')
    parser.add_argument('--beta', type=float, required=True,
                        help='Penalty weight.')
    return parser

def label_smooth(label, n_class=3, alpha=0.1):
    '''
    label: true label
    n_class: # of class
    alpha: smooth factor
    '''
    k = alpha / (n_class - 1)
    temp = torch.full((label.shape[0], n_class), k)
    temp = temp.scatter_(1, label.unsqueeze(1), (1-alpha))
    return temp

class LabelSmoothingLoss(nn.Module):
    def __init__(self, classes, smoothing=0.0, dim=-1):
        super(LabelSmoothingLoss, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.cls = classes
        self.dim = dim
        self.loss = nn.KLDivLoss()

    def forward(self, pred, target):
        pred = pred.log_softmax(dim = self.dim)
        with torch.no_grad():
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.smoothing/(self.cls-1))
            true_dist.scatter_(1, target.data.unsqueeze(1),self.confidence)
        loss = torch.mean(torch.sum(-true_dist*pred, dim=self.dim))
        loss = self.loss(pred, true_dist)
        return loss


class Derpp(ContinualModel):
    NAME = 'derpp'
    COMPATIBILITY = ['class-il', 'domain-il', 'task-il', 'general-continual']

    def __init__(self, backbone, loss, args, transform):
        super(Derpp, self).__init__(backbone, loss, args, transform)

        self.buffer = Buffer(self.args.buffer_size, self.device)
        self.label_smooth_loss = LabelSmoothingLoss(classes=10, smoothing=0.05)


    def begin_task(self, dataset):
        self.N_CLASSES_PER_TASK = dataset.N_CLASSES_PER_TASK
        # path = os.path.join(os.getcwd(), 'resnet18.pth')
        # if os.path.exists(path):
        #     self.net.load_state_dict(torch.load(path))

    # def end_task(self, dataset):
    #     dataset.task_id = dataset.i//dataset.N_CLASSES_PER_TASK 
    #     # task_id = dataset.i//dataset.N_CLASSES_PER_TASK 
    #     torch.save(self.net.state_dict(), 'resnet18.pth') 
    
    # classifier Reuse
    def end_task(self, dataset):
        dataset.task_id = dataset.i//dataset.N_CLASSES_PER_TASK 
        # Other classifier Reuse
        # task_id = dataset.i//dataset.N_CLASSES_PER_TASK 
        # torch.save(self.net.state_dict(), 'resnet18.pth') 
        # temp = task_id
        # print('temp',temp)
        # N = dataset.N_CLASSES_PER_TASK 
        # while temp != dataset.N_TASKS:
        #     self.net.fc.weight.data[N*temp:N*temp+N] = self.net.fc.weight.data[N*task_id-N:N*task_id]
        #     self.net.fc.bias.data[N*temp:N*temp+N] = self.net.fc.bias.data[N*task_id-N:N*task_id]
        #     temp += 1

    def compute_offsets(self, task):
        # mapping from classes [1-100] to their idx within a task
        offset1 = task * self.N_CLASSES_PER_TASK 
        offset2 = (task + 1) * self.N_CLASSES_PER_TASK 
        return int(offset1), int(offset2)       

    def take_multitask_loss(self, logits, y, t):
        loss = 0.0
        for i, _ in enumerate(y):
            offset1, offset2 = self.compute_offsets(t)
            offset2 = 200
            loss += self.loss(logits[i, offset1:offset2].unsqueeze(0), y[i].unsqueeze(0)-offset1)
        return loss/len(y)

    # mask_trick
    def take_multitask_loss1(self, t, bt, logits, buf_logits, y, type):
        if type == 'MSE':
            loss = 0.0
            for i, ti in enumerate(bt):
                offset1, offset2 = self.compute_offsets(ti)
                # loss += F.mse_loss(logits[i, offset1:offset2].unsqueeze(0), y[i].unsqueeze(0)-offset1)
                
                loss += F.mse_loss(logits[i, offset1:offset2].unsqueeze(0), buf_logits[i, offset1:offset2].unsqueeze(0))
            return loss/len(bt)
        else:
            loss = 0.0
            for i, ti in enumerate(bt):
                # offset1, offset2 = self.compute_offsets(ti)
                offset1, offset2 = self.compute_offsets(t)
                offset1 = 0
                loss += self.loss(logits[i, offset1:offset2].unsqueeze(0), y[i].unsqueeze(0)-offset1)
                # lam = torch.exp(torch.tensor(-ti/5)) # for cifar10
                # loss += lam*self.loss(logits[i, offset1:offset2].unsqueeze(0), y[i].unsqueeze(0)-offset1)
            return loss/len(bt)   

    def observe(self, inputs, labels, not_aug_inputs,t, test=False):
        # M = 10
        real_batch_size = inputs.shape[0]
        # real_batch_size = int(np.random.randint(1, min(M, real_batch_size), size=1))
        perm = torch.randperm(real_batch_size)
        inputs, labels = inputs[perm], torch.tensor(labels[perm],dtype=torch.long)
        not_aug_inputs = not_aug_inputs[perm]

        self.opt.zero_grad()
        outputs = self.net(inputs)
        loss = self.loss(outputs,labels)
        # loss = self.label_smooth_loss(outputs,labels)
        # loss = self.take_multitask_loss(outputs, labels, t)

        if not self.buffer.is_empty():
            buf_inputs, buf_labels, buf_logits, buf_task = self.buffer.get_data(
                self.args.minibatch_size, transform=self.transform)
            buf_outputs = self.net(buf_inputs)
            loss += self.args.alpha * F.mse_loss(buf_outputs, buf_logits)
            # loss += self.args.alpha * self.take_multitask_loss1(buf_task, buf_outputs, buf_logits, buf_labels, type='MSE')

            '——— ++ part: Add the label supervision to avoid the logits bias when task index suddenly change. (In this situation, ' \
            'the logits are more incline to the previous class)'
            buf_inputs, buf_labels, buf_logits, buf_task  = self.buffer.get_data(
                self.args.minibatch_size, transform=self.transform)
            # print('buf_inputs', buf_inputs.shape)
            buf_outputs = self.net(buf_inputs)
            loss += self.args.beta * self.loss(buf_outputs, buf_labels)
            # loss += self.args.beta * self.take_multitask_loss1(t, buf_task, buf_outputs, buf_logits, buf_labels, type='loss')
        loss.backward()
        self.opt.step()

        self.buffer.add_data(examples=not_aug_inputs,
                             labels=labels,
                             logits=outputs.data,
                             task_labels=t*torch.ones_like(labels))
        return loss.item()

    def observe_test(self, inputs, labels, test=False):
        self.opt.zero_grad()
        # outputs = self.net(inputs)
        outputs, fea = self.net(inputs, returnt='all')
        # torch.save(fea, 'derpp_fea.pt')
        loss = self.loss(outputs,labels)
        # loss = self.take_multitask_loss(outputs, labels, t)

        if not self.buffer.is_empty():
            buf_inputs, buf_labels, buf_logits, buf_task = self.buffer.get_data(
                self.args.minibatch_size, transform=self.transform)
            buf_outputs = self.net(buf_inputs)
            loss += self.args.alpha * F.mse_loss(buf_outputs, buf_logits)
            # loss += self.args.alpha * self.take_multitask_loss1(buf_task, buf_outputs, buf_logits, buf_labels, type='MSE')

            '——— ++ part: Add the label supervision to avoid the logits bias when task index suddenly change. (In this situation, ' \
            'the logits are more incline to the previous class)'
            buf_inputs, buf_labels, buf_logits, buf_task  = self.buffer.get_data(
                self.args.minibatch_size, transform=self.transform)
            buf_outputs = self.net(buf_inputs)
            loss += self.args.beta * self.loss(buf_outputs, buf_labels)
            # loss += self.args.beta * self.take_multitask_loss1(t, buf_task, buf_outputs, buf_logits, buf_labels, type='loss')
        loss.backward()
        self.opt.step()
        return loss.item() 

    # sgd
    # def observe_test(self, inputs, labels, ):
    #     real_batch_size = inputs.shape[0]
    #     perm = torch.randperm(real_batch_size)
    #     inputs, labels = inputs[perm], torch.tensor(labels[perm],dtype=torch.long)
    #     # not_aug_inputs = not_aug_inputs[perm]


    #     self.opt.zero_grad()
    #     outputs = self.net(inputs)
    #     # loss = self.take_multitask_loss(outputs, labels, t)
    #     loss = self.loss(outputs, labels)
    #     loss.backward()
    #     self.opt.step()

    #     return loss.item()