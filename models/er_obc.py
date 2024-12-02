# Copyright 2020-present, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Davide Abati, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# Copyright 2020-present, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Davide Abati, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
from functools import reduce
import torch
from utils.buffer import Buffer
from utils.ring_buffer import RingBuffer

from utils.args import *
from models.utils.continual_model import ContinualModel
from collections import OrderedDict
import torch.nn.functional as F
import numpy as np
import os
np.random.seed(0)
def get_parser() -> ArgumentParser:
    parser = ArgumentParser(description='Continual learning via'
                                        ' Experience Replay.')
    add_management_args(parser)
    add_experiment_args(parser)
    add_rehearsal_args(parser)
    parser.add_argument('--grad_clip_norm', type=float, help='learning rate', default=1.0)
    return parser

class ErOBC(ContinualModel):
    NAME = 'er_obc'
    COMPATIBILITY = ['class-il', 'domain-il', 'task-il', 'general-continual']

    def __init__(self, backbone, loss, args, transform):
        super(ErOBC, self).__init__(backbone, loss, args, transform)
        self.buffer = Buffer(self.args.buffer_size, self.device)
        self.iter = 0
        # self.buffer = RingBuffer(self.args.buffer_size, self.device, 5)
        # self.fast_weight = self.net.get_fast_weight()

    def begin_task(self, dataset):
        self.N_CLASSES_PER_TASK = dataset.N_CLASSES_PER_TASK
        self.N_TASKS = dataset.N_TASKS
        self.Total_classes = self.N_CLASSES_PER_TASK*self.N_TASKS
        # path = os.path.join(os.getcwd(), 'resnet18.pth')
        # if os.path.exists(path):
        #     self.net.load_state_dict(torch.load(path))

    def end_task(self, dataset):
        dataset.task_id = dataset.i//dataset.N_CLASSES_PER_TASK 
        # print('task_id', dataset.task_id)
        # Other classifier Reuse
        # task_id = dataset.i//dataset.N_CLASSES_PER_TASK 
        # torch.save(self.net.state_dict(), 'resnet18.pth') 

    def compute_offsets(self, task):
        # mapping from classes [1-100] to their idx within a task
        offset1 = task * self.N_CLASSES_PER_TASK 
        offset2 = (task + 1) * self.N_CLASSES_PER_TASK 
        # print('offset1', offset1)
        return int(offset1), int(offset2)                  

    def take_multitask_loss(self, bt, logits, y, task_id):
        loss = 0.0
        for i, ti in enumerate(bt):
            if i < 10:
                offset1, offset2 = self.compute_offsets(ti)
                offset2 = 10
            else:
                _, offset2 = self.compute_offsets(task_id)
                # task simiarity exp2
                offset1 = 0
            loss += self.loss(logits[i, offset1:offset2].unsqueeze(0), y[i].unsqueeze(0)-offset1)
        return loss/len(bt)


    def observe(self, inputs, labels, not_aug_inputs, task_id):
        real_batch_size = inputs.shape[0]
        perm = torch.randperm(real_batch_size)
        inputs, labels = inputs[perm], torch.tensor(labels[perm],dtype=torch.long)
        not_aug_inputs = not_aug_inputs[perm]

        self.opt.zero_grad()
        outputs = self.net(inputs, mode = 'surrogage')
        loss = self.loss(outputs, labels)
        if not self.buffer.is_empty():
            buf_inputs, buf_labels, buf_id = self.buffer.get_data(
                self.args.minibatch_size, transform=self.transform)
            loss += self.loss(self.net(buf_inputs), buf_labels)
        else:
            buf_id = task_id*torch.ones_like(labels)
        loss.backward()
        self.opt.step()
        self.opt.zero_grad()

        # update the balance classifier
        self.opt1.zero_grad()
        if not self.buffer.is_empty():
            buf_inputs, buf_labels, buf_id = self.buffer.get_data(
                50, transform=self.transform)           
            outputs = self.net(buf_inputs)
            loss = self.loss(outputs,buf_labels)
            loss.backward()
            self.opt1.step()
            self.opt1.zero_grad()

        self.buffer.add_data(examples=not_aug_inputs,
                             labels=labels,
                             task_labels=task_id*torch.ones_like(labels))

        return loss.item()

    def observe_test(self, inputs, labels, task_id):
        real_batch_size = inputs.shape[0]
        perm = torch.randperm(real_batch_size)
        inputs, labels = inputs[perm], torch.tensor(labels[perm],dtype=torch.long)

        self.opt.zero_grad()
        outputs = self.net(inputs)
        loss = self.loss(outputs, labels)
        if not self.buffer.is_empty():
            buf_inputs, buf_labels, buf_id = self.buffer.get_data(
                self.args.minibatch_size, transform=self.transform)
            loss += self.loss(self.net(buf_inputs), buf_labels)
        else:
            buf_id = task_id*torch.ones_like(labels)
        loss.backward()
        self.opt.step()
        self.opt.zero_grad()

        # update the balance classifier
        # self.opt1.zero_grad()
        # if not self.buffer.is_empty():
        #     buf_inputs, buf_labels, buf_id = self.buffer.get_data(
        #         16, transform=self.transform)           
        #     outputs = self.net(buf_inputs)
        #     loss = self.loss(outputs,buf_labels)
        #     loss.backward()
        #     self.opt1.step()
        #     self.opt1.zero_grad()

        return loss.item()




'''
    def observe(self, inputs, labels, not_aug_inputs, task_id):
        real_batch_size = inputs.shape[0]
        perm = torch.randperm(real_batch_size)
        inputs, labels = inputs[perm], torch.tensor(labels[perm],dtype=torch.long)
        not_aug_inputs = not_aug_inputs[perm]

        # self.opt.zero_grad()
        # outputs = self.net(inputs, mode = 'surrogage')
        # loss = self.loss(outputs, labels)
        # update the surrogate classifier
        self.opt.zero_grad()
        if not self.buffer.is_empty():
            buf_inputs, buf_labels, buf_id = self.buffer.get_data(
                self.args.minibatch_size, transform=self.transform)              
            inputs = torch.cat((inputs, buf_inputs))
            buf_id = torch.cat((task_id*torch.ones_like(labels),buf_id))
            labels = torch.cat((labels, buf_labels))
            # outputs = self.net(buf_inputs, mode = 'surrogage')
            # loss_buf = self.loss(outputs, buf_labels)
            # loss += loss_buf

        else:
            buf_id = task_id*torch.ones_like(labels)
        outputs = self.net(inputs, mode = 'surrogate')
        loss = self.loss(outputs, labels)
        # print('s_loss', loss)
        loss.backward()
        self.opt.step()
        self.opt.zero_grad()        

        # update the balance classifier
        self.opt1.zero_grad()
        if not self.buffer.is_empty():
            buf_inputs, buf_labels, buf_id = self.buffer.get_data(
                32, transform=self.transform)           
            outputs = self.net(buf_inputs)
            # buf_labels = F.one_hot(buf_labels, num_classes=  self.Total_classes)
            # smoothing = 0.1
            # buf_labels = (1-smoothing) *buf_labels + smoothing/(self.Total_classes)
            # loss = F.binary_cross_entropy_with_logits(outputs, buf_labels)
            loss = self.loss(outputs,buf_labels)
            # print('b_loss', loss)
            loss.backward()
            self.opt1.step()
            self.opt1.zero_grad()

        self.buffer.add_data(examples=not_aug_inputs,
                             labels=labels, 
                             task_labels=task_id*torch.ones_like(labels))
        return loss.item()

    def observe_test(self, inputs, labels, task_id):
        real_batch_size = inputs.shape[0]
        perm = torch.randperm(real_batch_size)
        inputs, labels = inputs[perm], torch.tensor(labels[perm],dtype=torch.long)

        self.opt.zero_grad()
        # outputs = self.net(inputs, mode = 'surrogage')
        # loss = self.loss(outputs, labels)
        # update the surrogate classifier
        self.opt.zero_grad()
        if not self.buffer.is_empty():
            buf_inputs, buf_labels, buf_id = self.buffer.get_data(
                self.args.minibatch_size, transform=self.transform)        
            inputs = torch.cat((inputs, buf_inputs))
            buf_id = torch.cat((task_id*torch.ones_like(labels),buf_id))
            labels = torch.cat((labels, buf_labels))      
            # outputs = self.net(buf_inputs, mode = 'surrogage')
            # loss_buf = self.loss(outputs, buf_labels)
            # loss += loss_buf

        else:
            buf_id = task_id*torch.ones_like(labels)
        outputs = self.net(inputs, mode = 'surrogate')
        loss = self.loss(outputs, labels)
        loss.backward()
        self.opt.step()
        self.opt.zero_grad()        

        # update the balance classifier
        self.opt1.zero_grad()
        if not self.buffer.is_empty():
            buf_inputs, buf_labels, buf_id = self.buffer.get_data(
                32, transform=self.transform)           
            outputs = self.net(buf_inputs)
            loss = self.loss(outputs,buf_labels)
            loss.backward()
            self.opt1.step()
            self.opt1.zero_grad()

        return loss.item()

'''

        
    # def observe_test(self, inputs, labels, task_id):
    #     real_batch_size = inputs.shape[0]
    #     perm = torch.randperm(real_batch_size)
    #     inputs, labels = inputs[perm], torch.tensor(labels[perm],dtype=torch.long)

    #     # update the surrogate classifier
    #     self.opt.zero_grad()
    #     if not self.buffer.is_empty():
    #         buf_inputs, buf_labels, buf_id = self.buffer.get_data(
    #             self.args.minibatch_size, transform=self.transform)              
    #         inputs = torch.cat((inputs, buf_inputs))
    #         buf_id = torch.cat((task_id*torch.ones_like(labels),buf_id))
    #         labels = torch.cat((labels, buf_labels))
    #     else:
    #         buf_id = task_id*torch.ones_like(labels)
    #     outputs = self.net(inputs, mode = 'surrogate')
    #     loss = self.loss(outputs, labels)
    #     loss.backward()
    #     self.opt.step()
    #     self.opt.zero_grad()        

    #     # update the balance classifier
    #     self.opt1.zero_grad()
    #     if not self.buffer.is_empty():
    #         buf_inputs, buf_labels, buf_id = self.buffer.get_data(
    #             50, transform=self.transform)           
    #         outputs = self.net(buf_inputs)
    #         # buf_labels = F.one_hot(buf_labels, num_classes=  self.Total_classes)
    #         # smoothing = 0.1
    #         # buf_labels = (1-smoothing) *buf_labels + smoothing/(self.Total_classes)
    #         # loss = F.binary_cross_entropy_with_logits(outputs, buf_labels)
    #         loss = self.loss(outputs,buf_labels)
    #         loss.backward()
    #         self.opt1.step()
    #         self.opt1.zero_grad()


    #     '''
    #     if self.iter %2 ==0:
    #         # for param in self.net.bn.paramters():
    #         #     param.requires_grad = False
    #         # for name, module in self.net.named_modules():
    #         #     if isinstance(module, torch.nn.BatchNorm2d):
    #         #         module.eval()
    #         #         for param in module.parameters():
    #         #             param.requires_grad = False

    #         self.opt.zero_grad()
    #         if not self.buffer.is_empty():
    #             buf_inputs, buf_labels, buf_id = self.buffer.get_data(
    #                 self.args.minibatch_size, transform=self.transform)              
    #             inputs = torch.cat((inputs, buf_inputs))
    #             buf_id = torch.cat((task_id*torch.ones_like(labels),buf_id))
    #             labels = torch.cat((labels, buf_labels))
    #         else:
    #             buf_id = task_id*torch.ones_like(labels)
    #         outputs = self.net(inputs, mode = 'surrogate')
    #         loss = self.loss(outputs, labels)
    #         loss.backward()
    #         self.opt.step()
    #         self.opt.zero_grad()
    #     else:
    #         # for name, module in self.net.named_modules():
    #         #     if isinstance(module, torch.nn.BatchNorm2d):
    #         #         module.train()
    #         #         for param in module.parameters():
    #         #             param.requires_grad = True
    #         self.opt1.zero_grad()
    #         if not self.buffer.is_empty():
    #             buf_inputs, buf_labels, buf_id = self.buffer.get_data(
    #                 32, transform=self.transform)           # 20 s=0.1 51% 52.13% s=0.05 47.38% s=0.15 48.82%  10 s=0.1    30 s=0.1  50.7%
    #         else:
    #             buf_id = task_id*torch.ones_like(labels)
    #         outputs = self.net(buf_inputs)
    #         buf_labels = F.one_hot(buf_labels, num_classes=  self.Total_classes)
    #         smoothing = 0
    #         buf_labels = (1-smoothing) *buf_labels + smoothing/(self.Total_classes)
    #         loss = F.binary_cross_entropy_with_logits(outputs, buf_labels)
    #         # loss = self.loss(outputs,buf_labels)
    #         loss.backward()
    #         self.opt1.step()
    #         self.opt1.zero_grad()
    #     self.iter += 1
    #     '''
            
    #     # self.buffer.add_data(examples=not_aug_inputs,
    #     #                      labels=labels, 
    #     #                      task_labels=task_id*torch.ones_like(labels))
        
       
    #     return loss.item()

