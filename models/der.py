# Copyright 2020-present, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Davide Abati, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from utils.buffer import Buffer
from torch.nn import functional as F
from utils.args import *   # introduce the parameters
from models.utils.continual_model import ContinualModel
import numpy as np
import torch

# 针对不同的模型, 需要引入的新的参数
def get_parser() -> ArgumentParser:
    parser = ArgumentParser(description='Continual learning via'
                                        ' Dark Experience Replay.')
    # args three types parameters
    add_management_args(parser)
    add_experiment_args(parser)
    add_rehearsal_args(parser)
    parser.add_argument('--alpha', type=float, required=True, # 这里的alpha是distillation loss的比重
                        help='Penalty weight.')
    return parser

class Der(ContinualModel):
    NAME = 'der'
    COMPATIBILITY = ['class-il', 'domain-il', 'task-il', 'general-continual']

    def __init__(self, backbone, loss, args, transform):
        super(Der, self).__init__(backbone, loss, args, transform)
        self.buffer = Buffer(self.args.buffer_size, self.device)


    def begin_task(self, dataset):
        self.N_CLASSES_PER_TASK = dataset.N_CLASSES_PER_TASK

    def end_task(self, dataset):
        dataset.task_id = dataset.i//dataset.N_CLASSES_PER_TASK 
        # print('task_id', dataset.task_id)
        # Other classifier Reuse
        task_id = dataset.i//dataset.N_CLASSES_PER_TASK 
        # torch.save(self.net.state_dict(), 'resnet18.pth') 
        # temp = task_id
        # # print('temp',temp)
        # N = dataset.N_CLASSES_PER_TASK 
        # while temp !=dataset.N_TASKS:
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
            offset1 = 0
            # lam = torch.exp(torch.tensor(-t/5)) # for cifar10
            # loss += lam*self.loss(logits[i, offset1:offset2].unsqueeze(0), y[i].unsqueeze(0)-offset1)
            loss += self.loss(logits[i, offset1:offset2].unsqueeze(0), y[i].unsqueeze(0)-offset1)
        return loss/len(y)


    def observe(self, inputs, labels, not_aug_inputs, t):
        # M = 10
        real_batch_size = inputs.shape[0]
        # real_batch_size = int(np.random.randint(1, min(M, real_batch_size), size=1))
        perm = torch.randperm(real_batch_size)
        inputs, labels = inputs[perm], torch.tensor(labels[perm],dtype=torch.long)
        not_aug_inputs = not_aug_inputs[perm]

        
        self.opt.zero_grad()
        outputs = self.net(inputs)
        loss = self.loss(outputs, labels)
        # loss = self.take_multitask_loss(outputs, labels, t)
        # 根据der的代码, minibatch_size = batch_size(# of not_aug_inputs) --> 第2个task  buffer(128个task 1)  第3个task buffer(~64个task2 ~64个task1)
        if not self.buffer.is_empty(): # buffer非空, 执行下面命令
            buf_inputs, buf_logits = self.buffer.get_data( # 为啥get_data还有logts 是谁的logits, 要看下line 46 add_data (解决！)
                self.args.minibatch_size, transform=self.transform)
            buf_outputs = self.net(buf_inputs)
            loss += self.args.alpha * F.mse_loss(buf_outputs, buf_logits)  # distillation loss according to the DER paper

        loss.backward()
        self.opt.step()
        self.buffer.add_data(examples=not_aug_inputs, logits=outputs.data)

        return loss.item()

    # def observe_test(self, inputs, labels ):
    #     # M = 10
    #     real_batch_size = inputs.shape[0]
    #     # real_batch_size = int(np.random.randint(1, min(M, real_batch_size), size=1))
    #     perm = torch.randperm(real_batch_size)
    #     inputs, labels = inputs[perm], torch.tensor(labels[perm],dtype=torch.long)
    #     # not_aug_inputs = not_aug_inputs[perm]

        
    #     self.opt.zero_grad()
    #     outputs = self.net(inputs)
    #     loss = self.loss(outputs, labels)
    #     # loss = self.take_multitask_loss(outputs, labels, t)
    #     # 根据der的代码, minibatch_size = batch_size(# of not_aug_inputs) --> 第2个task  buffer(128个task 1)  第3个task buffer(~64个task2 ~64个task1)
    #     if not self.buffer.is_empty(): # buffer非空, 执行下面命令
    #         buf_inputs, buf_logits = self.buffer.get_data( # 为啥get_data还有logts 是谁的logits, 要看下line 46 add_data (解决！)
    #             self.args.minibatch_size, transform=self.transform)
    #         buf_outputs = self.net(buf_inputs)
    #         loss += self.args.alpha * F.mse_loss(buf_outputs, buf_logits)  # distillation loss according to the DER paper

    #     loss.backward()
    #     self.opt.step()
    #     # self.buffer.add_data(examples=inputs, logits=outputs.data)

    #     return loss.item()

    # sgd
    def observe_test(self, inputs, labels, ):
        real_batch_size = inputs.shape[0]
        perm = torch.randperm(real_batch_size)
        inputs, labels = inputs[perm], torch.tensor(labels[perm],dtype=torch.long)
        # not_aug_inputs = not_aug_inputs[perm]


        self.opt.zero_grad()
        outputs = self.net(inputs)
        # loss = self.take_multitask_loss(outputs, labels, t)
        loss = self.loss(outputs, labels)
        loss.backward()
        self.opt.step()

        return loss.item()
