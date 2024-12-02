# Copyright 2020-present, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Davide Abati, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.args import *
from models.utils.continual_model import ContinualModel
import numpy as np
from copy import deepcopy

def get_parser() -> ArgumentParser:
    parser = ArgumentParser(description='Continual learning via'
                                        ' online EWC.')
    add_management_args(parser)
    add_experiment_args(parser)
    parser.add_argument('--e_lambda', type=float, required=True,
                        help='lambda weight for EWC')
    parser.add_argument('--gamma', type=float, required=True,
                        help='gamma parameter for EWC online')

    return parser


class EwcOn(ContinualModel):
    NAME = 'ewc_on'
    COMPATIBILITY = ['class-il', 'domain-il', 'task-il']

    def __init__(self, backbone, loss, args, transform):
        super(EwcOn, self).__init__(backbone, loss, args, transform)

        self.logsoft = nn.LogSoftmax(dim=1)
        self.checkpoint = None
        self.fish = None
        self.grad1, self.var1 = [],[]
        self.iteration = 0
        self.p,self.p_last = 0,0
        self.distance = []
        self.grad=[]
        self.t =0
        self.task_id = -1
    def penalty(self):
        if self.checkpoint is None:
            return torch.tensor(0.0).to(self.device)
        else:
            penalty = (self.fish * ((self.net.get_params() - self.checkpoint) ** 2)).sum()
            return penalty

    def begin_task(self, dataset):
        self.N_CLASSES_PER_TASK = dataset.N_CLASSES_PER_TASK


    def end_task(self, dataset):
        fish = torch.zeros_like(self.net.get_params())
        for j, data in enumerate(dataset.train_loader):
            inputs, labels, _ = data
            inputs, labels = inputs.to(self.device), torch.tensor(labels.to(self.device),dtype=torch.long)
            for ex, lab in zip(inputs, labels):
                self.opt.zero_grad()
                output = self.net(ex.unsqueeze(0))
                loss = - F.nll_loss(self.logsoft(output), lab.unsqueeze(0),
                                    reduction='none')
                exp_cond_prob = torch.mean(torch.exp(loss.detach().clone()))
                loss = torch.mean(loss)
                loss.backward()
                fish += exp_cond_prob * self.net.get_grads() ** 2

        fish /= (len(dataset.train_loader) * self.args.batch_size)

        if self.fish is None:
            self.fish = fish
        else:
            self.fish *= self.args.gamma
            self.fish += fish
        # torch.save(self.fish,'fisher.pt') 


        # self.checkpoint = self.net.get_params().data.clone()
        # dataset.task_id = dataset.i//dataset.N_CLASSES_PER_TASK 
        # task_id = dataset.i//dataset.N_CLASSES_PER_TASK 
        # torch.save(self.net.state_dict(), 'resnet18.pth') 
        # temp = task_id
        # # print('temp',temp)
        # N = dataset.N_CLASSES_PER_TASK 
        # while temp !=dataset.N_TASKS:
        #     self.net.fc.weight.data[N*temp:N*temp+N] = self.net.fc.weight.data[N*task_id-N:N*task_id]
        #     self.net.fc.bias.data[N*temp:N*temp+N] = self.net.fc.bias.data[N*task_id-N:N*task_id]
        #     temp += 1

    # original
    # def end_task(self, dataset):
    #     fish = torch.zeros_like(self.net.get_params())
    #     # print('iteration',self.iteration)
    #     # diff_list = []
    #     # for i in range(len(self.distance)):
    #     #     for j in range(len(self.distance)):
    #     #         # diff_list.append(torch.norm(self.distance[i]-self.distance[j]))
    #     #         diff_list.append(1-F.cosine_similarity(self.distance[i].unsqueeze(0),self.distance[j].unsqueeze(0)))
    #     # print('\n distance_matrix', torch.stack(diff_list))
    #     for j, data in enumerate(dataset.train_loader):
    #         inputs, labels, _ = data
    #         inputs, labels = inputs.to(self.device), torch.tensor(labels.to(self.device),dtype=torch.long)
    #         for ex, lab in zip(inputs, labels):
    #             self.opt.zero_grad()
    #             output = self.net(ex.unsqueeze(0))
    #             loss = - F.nll_loss(self.logsoft(output), lab.unsqueeze(0),
    #                                 reduction='none')
    #             exp_cond_prob = torch.mean(torch.exp(loss.detach().clone()))
    #             loss = torch.mean(loss)
    #             loss.backward()
    #             fish += exp_cond_prob * self.net.get_grads() ** 2
    #     fish /= (len(dataset.train_loader) * self.args.batch_size)

    #     if self.fish is None:
    #         self.fish = fish
    #     else:
    #         self.fish *= self.args.gamma
    #         self.fish += fish
    #     # torch.save(self.fish,'fisher.pt') 
    #     self.checkpoint = self.net.get_params().data.clone()


    def compute_offsets(self, task):
        # mapping from classes [1-100] to their idx within a task
        offset1 = task * self.N_CLASSES_PER_TASK 
        offset2 = (task + 1) * self.N_CLASSES_PER_TASK 
        return int(offset1), int(offset2)       
            
    def take_multitask_loss(self, logits, y, t):
        loss = 0.0
        for i, ti in enumerate(y):
            offset1, offset2 = self.compute_offsets(t)
            offset1 = 0
            loss += self.loss(logits[i, offset1:offset2].unsqueeze(0), y[i].unsqueeze(0)-offset1)
        return loss/len(y)

    def observe(self, inputs, labels, not_aug_inputs, t):
        if t != self.task_id:
            # self.grad1= []
            self.iteration = 0
            self.task_id = t

        
        real_batch_size = inputs.shape[0]
        perm = torch.randperm(real_batch_size)
        inputs, labels = inputs[perm], torch.tensor(labels[perm],dtype=torch.long)
        not_aug_inputs = not_aug_inputs[perm]

        self.opt.zero_grad()
        outputs = self.net(inputs)
        penalty = self.penalty()
        loss = self.loss(outputs, labels) + self.args.e_lambda * penalty
        # loss = self.take_multitask_loss(outputs, labels, t) + self.args.e_lambda * penalty
        assert not torch.isnan(loss)
        loss.backward()
        cnt = 0
        # for p in self.net.parameters():
            # if  cnt == 3 and self.iter>3000:
            # if cnt ==3:
            #     if self.p==0 or t==0:
            #         self.p_last = deepcopy(p.data.flatten())
            #         self.p=1
            #     else:
            #         self.distance.append(torch.norm(p.data.flatten()- self.p_last))
            #         torch.save(torch.stack(self.distance),'params_dis.pt')
            #         # self.p_last = deepcopy(p.data.flatten())
            #     break

            # if cnt == 3:# and self.iteration<500:d
            #     # print("!!!!!!!!!!")
            #     # self.grad1.append(deepcopy(p.grad.flatten()))
            #     # norm1 = torch.norm(torch.mean(torch.stack(self.grad1),dim=0),2)
            #     # self.var1.append((torch.var(torch.stack(self.grad1),dim=0)/norm1).sum())
            #     # torch.save(torch.stack(self.var1),'var_ewc_imb.pt') 

            #     temp = deepcopy(p.grad.flatten())
            #     temp =  torch.norm(temp,p=2)
            #     self.grad.append(temp)
            #     norm = torch.mean(torch.stack(self.grad))

            #     self.grad1.append(deepcopy(p.grad.flatten()))
            #     norm1 = torch.norm(torch.mean(torch.stack(self.grad1),dim=0),2)
            #     self.var1.append((norm-norm1)/norm1)
            #     torch.save(self.var1, 'var_ewc_online.pt')   

            # if cnt ==9:
            #     if  self.t==t:
            #         if len(self.distance) == 0:
            #             self.distance.append(deepcopy(p.data.flatten()))
            #         else:
            #             self.distance[-1] = deepcopy(p.data.flatten())
            #     else:
            #         self.t +=1
            #         self.distance.append(deepcopy(p.data.flatten()))                   
            # cnt += 1
        # print('cnt',cnt)
        self.opt.step()
        self.iteration += 1
        return loss.item()

    def observe_test(self, inputs, labels):
        real_batch_size = inputs.shape[0]
        perm = torch.randperm(real_batch_size)
        inputs, labels = inputs[perm], torch.tensor(labels[perm],dtype=torch.long)

        self.opt.zero_grad()
        outputs = self.net(inputs)
        penalty = self.penalty()
        loss = self.loss(outputs, labels) + self.args.e_lambda * penalty
        # loss = self.take_multitask_loss(outputs, labels, t) + self.args.e_lambda * penalty
        assert not torch.isnan(loss)
        loss.backward()
        cnt = 0
        self.opt.step()
        self.iteration += 1
        return loss.item()
    
    #sgd
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
