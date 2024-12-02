# Copyright 2022-present, Lorenzo Bonicelli, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import numpy as np
from utils.buffer import Buffer
from models.gem import overwrite_grad
from models.gem import store_grad
from utils.args import *
from models.utils.continual_model import ContinualModel

def get_parser() -> ArgumentParser:
    parser = ArgumentParser(description='Continual learning via A-GEM.')
    add_management_args(parser)
    add_experiment_args(parser)
    add_rehearsal_args(parser)
    return parser

def project(gxy: torch.Tensor, ger: torch.Tensor) -> torch.Tensor:
    corr = torch.dot(gxy, ger) / torch.dot(ger, ger)
    return gxy - corr * ger


class AGem(ContinualModel):
    NAME = 'agem'
    COMPATIBILITY = ['class-il', 'domain-il', 'task-il']

    def __init__(self, backbone, loss, args, transform):
        super(AGem, self).__init__(backbone, loss, args, transform)

        self.buffer = Buffer(self.args.buffer_size, self.device)
        self.grad_dims = []
        for param in self.parameters():
            self.grad_dims.append(param.data.numel())
        self.grad_xy = torch.Tensor(np.sum(self.grad_dims)).to(self.device)
        self.grad_er = torch.Tensor(np.sum(self.grad_dims)).to(self.device)

    def begin_task(self, dataset):
        self.N_CLASSES_PER_TASK = dataset.N_CLASSES_PER_TASK

    # def end_task(self, dataset):
    #     samples_per_task = self.args.buffer_size // dataset.N_TASKS
    #     loader = dataset.train_loader
    #     cur_y, cur_x = next(iter(loader))[1:]
    #     self.buffer.add_data(
    #         examples=cur_x.to(self.device),
    #         labels=cur_y.to(self.device)
    #     )

    def end_task(self, dataset):
        dataset.task_id = dataset.i//dataset.N_CLASSES_PER_TASK 
        # Other classifier Reuse
        task_id = dataset.i//dataset.N_CLASSES_PER_TASK 
        torch.save(self.net.state_dict(), 'resnet18.pth') 
        temp = task_id
        # print('temp',temp)
        # N = dataset.N_CLASSES_PER_TASK 
        # while temp !=dataset.N_TASKS:
        #     self.net.fc.weight.data[N*temp:N*temp+N] = self.net.fc.weight.data[N*task_id-N:N*task_id]
        #     self.net.fc.bias.data[N*temp:N*temp+N] = self.net.fc.bias.data[N*task_id-N:N*task_id]
        #     temp += 1
        # samples_per_task = self.args.buffer_size // dataset.N_TASKS
        # loader = dataset.train_loader
        # cur_y, cur_x = next(iter(loader))[1:]
        # self.buffer.add_data(
        #     examples=cur_x.to(self.device),
        #     labels=cur_y.to(self.device)
        # )

    def compute_offsets(self, task):
        # mapping from classes [1-100] to their idx within a task
        offset1 = task * self.N_CLASSES_PER_TASK 
        offset2 = (task + 1) * self.N_CLASSES_PER_TASK 
        return int(offset1), int(offset2)       
            
    # def take_multitask_loss(self, bt, logits, y, task_id):
    #     loss = 0.0
    #     # print('bt',bt)
    #     # print('y',y)
    #     for i, ti in enumerate(bt):
    #         if ti == task_id:
    #             offset1, offset2 = self.compute_offsets(ti)
    #         else:
    #             _, offset2 = self.compute_offsets(task_id)
    #             offset1 = 0
    #         # print('offset1',offset1)
    #         # loss += self.loss(logits[i, offset1:offset2].unsqueeze(0), y[i].unsqueeze(0)-offset1)
    #         lam = torch.exp(torch.tensor(-ti/5)) # for cifar10
    #         loss += lam*self.loss(logits[i, offset1:offset2].unsqueeze(0), y[i].unsqueeze(0)-offset1)
    #     return loss/len(bt)

    def take_multitask_loss(self, logits, y, t):
        loss = 0.0
        for i, ti in enumerate(y):
            offset1, offset2 = self.compute_offsets(t)
            loss += self.loss(logits[i, offset1:offset2].unsqueeze(0), y[i].unsqueeze(0)-offset1)
        return loss/len(y)

    def observe(self, inputs, labels, not_aug_inputs, t):
        # M = 10
        real_batch_size = inputs.shape[0]
        # real_batch_size = int(np.random.randint(1, min(M, real_batch_size), size=1))
        # real_batch_size  = int(np.random.normal(5,2,1))
        # real_batch_size = max(1,real_batch_size)
        # real_batch_size = min(10, real_batch_size)
        # print('real_batch_size',real_batch_size)
        perm = torch.randperm(real_batch_size)
        # print('inputs.shape', inputs.shape)
        # print('perm', perm)
        inputs, labels = inputs[perm], torch.tensor(labels[perm],dtype=torch.long)
        # print('inputs.shape', inputs.shape)
        # print('targets.shape', labels.shape)
        not_aug_inputs = not_aug_inputs[perm]

        self.zero_grad()
        p = self.net.forward(inputs)
        loss = self.loss(p, labels)
        # buf_id = t*torch.ones_like(labels)
        # loss = self.take_multitask_loss(buf_id, p, labels, t)
        # loss = self.take_multitask_loss(p, labels, t)

        loss.backward()

        if not self.buffer.is_empty():
            store_grad(self.parameters, self.grad_xy, self.grad_dims)

            buf_inputs, buf_labels = self.buffer.get_data(self.args.minibatch_size, transform=self.transform)
            self.net.zero_grad()
            buf_outputs = self.net.forward(buf_inputs)
            penalty = self.loss(buf_outputs, buf_labels)
            penalty.backward()
            store_grad(self.parameters, self.grad_er, self.grad_dims)

            dot_prod = torch.dot(self.grad_xy, self.grad_er)
            if dot_prod.item() < 0:
                g_tilde = project(gxy=self.grad_xy, ger=self.grad_er)
                overwrite_grad(self.parameters, g_tilde, self.grad_dims)
            else:
                overwrite_grad(self.parameters, self.grad_xy, self.grad_dims)

        self.opt.step()
        self.buffer.add_data(examples=not_aug_inputs,
                             labels=labels)
        return loss.item()

    # def observe_test(self, inputs, labels):
    #     # M = 10
    #     real_batch_size = inputs.shape[0]
    #     # real_batch_size = int(np.random.randint(1, min(M, real_batch_size), size=1))
    #     # real_batch_size  = int(np.random.normal(5,2,1))
    #     # real_batch_size = max(1,real_batch_size)
    #     # real_batch_size = min(10, real_batch_size)
    #     # print('real_batch_size',real_batch_size)
    #     perm = torch.randperm(real_batch_size)
    #     # print('inputs.shape', inputs.shape)
    #     # print('perm', perm)
    #     inputs, labels = inputs[perm], torch.tensor(labels[perm],dtype=torch.long)
    #     # print('inputs.shape', inputs.shape)
    #     # print('targets.shape', labels.shape)
    #     # not_aug_inputs = not_aug_inputs[perm]

    #     self.zero_grad()
    #     p = self.net.forward(inputs)
    #     loss = self.loss(p, labels)
    #     # buf_id = t*torch.ones_like(labels)
    #     # loss = self.take_multitask_loss(buf_id, p, labels, t)
    #     # loss = self.take_multitask_loss(p, labels, t)

    #     loss.backward()
    #     self.opt.step()

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
