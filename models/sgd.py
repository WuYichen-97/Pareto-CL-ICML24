# Copyright 2020-present, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Davide Abati, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from utils.args import *
from models.utils.continual_model import ContinualModel
import numpy as np
import torch

def get_parser() -> ArgumentParser:
    parser = ArgumentParser(description='Continual Learning via'
                                        ' Progressive Neural Networks.')
    add_management_args(parser)
    add_experiment_args(parser)
    return parser


class Sgd(ContinualModel):
    NAME = 'sgd'
    COMPATIBILITY = ['class-il', 'domain-il', 'task-il', 'general-continual']

    def __init__(self, backbone, loss, args, transform):
        super(Sgd, self).__init__(backbone, loss, args, transform)
    def begin_task(self, dataset):
        self.N_CLASSES_PER_TASK = dataset.N_CLASSES_PER_TASK

    def compute_offsets(self, task):
        # mapping from classes [1-100] to their idx within a task
        offset1 = task * self.N_CLASSES_PER_TASK
        offset2 = (task + 1) * self.N_CLASSES_PER_TASK
        return int(offset1), int(offset2)  

    # def end_task(self, dataset):
    #     dataset.task_id = dataset.i//dataset.N_CLASSES_PER_TASK 
    #     # print('task_id', dataset.task_id)
    #     # Other classifier Reuse
    #     task_id = dataset.i//dataset.N_CLASSES_PER_TASK 
    #     torch.save(self.net.state_dict(), 'resnet18.pth') 
    #     temp = task_id
    #     N = dataset.N_CLASSES_PER_TASK 
    #     # print('temp',temp)
    #     while temp !=dataset.N_TASKS:
    #         self.net.fc.weight.data[N*temp:N*temp+N] = self.net.fc.weight.data[N*task_id-N:N*task_id]
    #         self.net.fc.bias.data[N*temp:N*temp+N] = self.net.fc.bias.data[N*task_id-N:N*task_id]
    #         temp += 1


    def take_multitask_loss(self, logits, y, t):
        loss = 0.0
        for i, _ in enumerate(y):
            offset1, offset2 = self.compute_offsets(t)
            offset2 = 10
            loss += self.loss(logits[i, offset1:offset2].unsqueeze(0), y[i].unsqueeze(0)-offset1)
        return loss/len(y)

    def observe(self, inputs, labels, not_aug_inputs,t):
        real_batch_size = inputs.shape[0]
        perm = torch.randperm(real_batch_size)
        inputs, labels = inputs[perm], torch.tensor(labels[perm],dtype=torch.long)
        not_aug_inputs = not_aug_inputs[perm]


        self.opt.zero_grad()
        outputs = self.net(inputs)
        # loss = self.take_multitask_loss(outputs, labels, t)
        loss = self.loss(outputs, labels)
        loss.backward()
        self.opt.step()

        return loss.item()
        
    def observe_test(self, inputs, labels, ):
        real_batch_size = inputs.shape[0]
        perm = torch.randperm(real_batch_size)
        inputs, labels = inputs[perm], torch.tensor(labels[perm],dtype=torch.long)
        # not_aug_inputs = not_aug_inputs[perm]


        self.opt.zero_grad()
        # outputs = self.net(inputs)
        outputs, fea = self.net(inputs, returnt='all')
        torch.save(fea, 'sgd_fea.pt')
        # loss = self.take_multitask_loss(outputs, labels, t)
        loss = self.loss(outputs, labels)
        loss.backward()
        self.opt.step()

        return loss.item()


# # Copyright 2020-present, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Davide Abati, Simone Calderara.
# # All rights reserved.
# # This source code is licensed under the license found in the
# # LICENSE file in the root directory of this source tree.

# from utils.args import *
# from models.utils.continual_model import ContinualModel
# import numpy as np
# import torch
# from copy import deepcopy

# def get_parser() -> ArgumentParser:
#     parser = ArgumentParser(description='Continual Learning via'
#                                         ' Progressive Neural Networks.')
#     add_management_args(parser)
#     add_experiment_args(parser)
#     return parser


# class Sgd(ContinualModel):
#     NAME = 'sgd'
#     COMPATIBILITY = ['class-il', 'domain-il', 'task-il', 'general-continual']

#     def __init__(self, backbone, loss, args, transform):
#         super(Sgd, self).__init__(backbone, loss, args, transform)
#         self.momentum = [[None]]*12
#         self.accumu_grad = [[None]]*12
#         self.grad = [[None]]*12
#         self.grad1 = []
#         self.var1 = []

#     def begin_task(self, dataset):
#         self.N_CLASSES_PER_TASK = dataset.N_CLASSES_PER_TASK
#         self.fast_weight = self.net.get_fast_weight()

#     def compute_offsets(self, task):
#         # mapping from classes [1-100] to their idx within a task
#         offset1 = task * self.N_CLASSES_PER_TASK 
#         offset2 = (task + 1) * self.N_CLASSES_PER_TASK 
#         return int(offset1), int(offset2)       
            
#     def take_multitask_loss(self, bt, logits, y):
#         loss = 0.0
#         for i, ti in enumerate(bt):
#             offset1, offset2 = self.compute_offsets(ti)
#             loss += self.loss(logits[i, offset1:offset2].unsqueeze(0), y[i].unsqueeze(0)-offset1)
#         return loss/len(bt)


#     def adp_lr_a(self, x):
#         scaling = torch.pow((0.1 + x),1/3)
#         lr = 0.1/scaling
#         a = 100*lr**2
#         return lr, min(a,1)

#     def inner_loop(self, x, y, t):       
#         # logits = self.net.functional_forward(x, fast_weight=self.fast_weight)
#         logits = self.net(x)
#         loss = self.take_multitask_loss(t,logits,y)
#         loss.backward()

#         cnt = 0
#         for p in self.net.parameters():
#             # p.data = p.data - (p.grad) * self.args.lr

#             if self.momentum[cnt][0] == None:
#                 self.momentum[cnt] = [p.grad]
#                 self.accumu_grad[cnt] = torch.pow(torch.norm(p.grad.flatten()), 2)
#                 self.grad[cnt] = [p.grad]
#                 p.data = p.data - (p.grad) * self.args.lr

#             else:
#                 self.accumu_grad[cnt]  += torch.pow(torch.norm(p.grad.flatten()), 2)
#                 lr, a = self.adp_lr_a(self.accumu_grad[cnt])
                
#                 self.momentum[cnt] = [p.grad + (1-a)*(self.momentum[cnt][0]-self.grad[cnt][0])]
#                 self.grad[cnt] = [p.grad]
#                 p.data = p.data - (self.momentum[cnt][0]) * lr 

#             if cnt == 3:
#                 self.grad1.append(deepcopy(self.momentum[cnt][0].flatten())) 
#                 # self.grad1.append(deepcopy(p.grad.flatten()))
#                 # if len(self.grad1)==50:
#                 #     self.grad1.pop(0)
#                 # self.grad1.append(deepcopy(p.grad.flatten()))
#                 # self.var.append(torch.var(torch.stack(self.grad)))
#                 self.var1.append(torch.var(torch.stack(self.grad1), dim=0).sum())
#                 # print('var',  torch.var(torch.stack(self.grad1)))
#                 # print('p', p.grad.flatten())
#                 # print('grad', torch.stack(self.grad1))
#                 # print('var', torch.var(torch.stack(self.grad1), dim=0).sum())
#                 # print('var', self.var)
#                 # print('var1', self.var1)
#                 # torch.save(torch.stack(self.var), 'var.pt')
#                 # torch.save(torch.stack(self.grad1),'grad1.pt')
#                 # torch.save(torch.stack(self.var1), 'var1.pt')

#                 # torch.save(torch.stack(self.grad1),'grad.pt')
#                 torch.save(torch.stack(self.var1), 'var-sgd-storm.pt')
#             cnt += 1
#         # self.fast_weight = self.net.get_fast_weight()
#         return loss
        



#     def observe(self, inputs, labels, not_aug_inputs,t):
#         real_batch_size = inputs.shape[0]
#         perm = torch.randperm(real_batch_size)
#         inputs, labels = inputs[perm], torch.tensor(labels[perm],dtype=torch.long)
#         not_aug_inputs = not_aug_inputs[perm]
#         buf_id = t*torch.ones_like(labels)

#         self.opt.zero_grad()
#         # outputs = self.net(inputs)
#         x_s, y_s = inputs, torch.tensor(labels, dtype=torch.long)
#         loss = self.inner_loop(x_s, y_s, buf_id)  
#         # one step adaptation

#         # loss = self.take_multitask_loss(buf_id, outputs, labels)
#         # loss.backward()
#         # self.opt.step()

#         return loss.item()
