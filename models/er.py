
import torch
from utils.buffer import Buffer
from utils.args import *
from models.utils.continual_model import ContinualModel
import numpy as np
np.random.seed(0)
import os

def get_parser() -> ArgumentParser:
    parser = ArgumentParser(description='Continual learning via'
                                        ' Experience Replay.')
    add_management_args(parser)
    add_experiment_args(parser)
    add_rehearsal_args(parser)
    return parser


class Er(ContinualModel):
    NAME = 'er'
    COMPATIBILITY = ['class-il', 'domain-il', 'task-il', 'general-continual']

    def __init__(self, backbone, loss, args, transform):
        super(Er, self).__init__(backbone, loss, args, transform)
        self.buffer = Buffer(self.args.buffer_size, self.device)

    # def begin_task(self, dataset):
    #     self.N_CLASSES_PER_TASK = dataset.N_CLASSES_PER_TASK

    def begin_task(self, dataset):
        self.N_CLASSES_PER_TASK = dataset.N_CLASSES_PER_TASK
        # path = os.path.join(os.getcwd(), 'resnet18.pth')
        # if os.path.exists(path):
        #     self.net.load_state_dict(torch.load(path))
   

 
    def end_task(self, dataset):
        dataset.task_id = dataset.i//dataset.N_CLASSES_PER_TASK 
        # Other classifier Reuse
        # task_id = dataset.i//dataset.N_CLASSES_PER_TASK 
        # torch.save(self.net.state_dict(), 'resnet18.pth') 
    #     temp = task_id
    #     # print('temp',temp)
    #     N = dataset.N_CLASSES_PER_TASK 
    #     while temp != dataset.N_TASKS:
    #         self.net.fc.weight.data[N*temp:N*temp+N-1] = self.net.fc.weight.data[N*task_id-2:N*task_id-1]
    #         self.net.fc.weight.data[N*temp+1:N*temp+N] = self.net.fc.weight.data[N*task_id-1:N*task_id]
    #         self.net.fc.bias.data[N*temp:N*temp+N-1] = self.net.fc.bias.data[N*task_id-2:N*task_id-1]
    #         self.net.fc.bias.data[N*temp+1:N*temp+N] = self.net.fc.bias.data[N*task_id-1:N*task_id]
    #         temp += 1


    def compute_offsets(self, task):
        # mapping from classes [1-100] to their idx within a task
        offset1 = task * self.N_CLASSES_PER_TASK
        offset2 = (task + 1) * self.N_CLASSES_PER_TASK
        return int(offset1), int(offset2)       
            
    # def take_multitask_loss(self, bt, logits, y):
    #     loss = 0.0
    #     for i, ti in enumerate(bt):
    #         offset1, offset2 = self.compute_offsets(ti)
    #         loss += self.loss(logits[i, offset1:offset2].unsqueeze(0), y[i].unsqueeze(0)-offset1)
    #     return loss/len(bt)

    def take_multitask_loss(self, bt, logits, y, task_id):
        loss = 0.0
        for i, ti in enumerate(bt):
            if i < 32:
                offset1, offset2 = self.compute_offsets(ti)
                offset2 = 10
                loss += self.loss(logits[i, offset1:offset2].unsqueeze(0), y[i].unsqueeze(0)-offset1)
            else:
                _, offset2 = self.compute_offsets(task_id)
                offset1 = 0
                loss += 0.5*self.loss(logits[i, offset1:offset2].unsqueeze(0), y[i].unsqueeze(0)-offset1)
        return loss/len(bt)


    def observe(self, inputs, labels, not_aug_inputs, task_id):
        real_batch_size = inputs.shape[0]
        perm = torch.randperm(real_batch_size)
        inputs, labels = inputs[perm], torch.tensor(labels[perm],dtype=torch.long)
        not_aug_inputs = not_aug_inputs[perm]

        self.opt.zero_grad()
        outputs = self.net(inputs)
        loss = self.loss(outputs, labels)
        if not self.buffer.is_empty():
            buf_inputs, buf_labels, buf_id = self.buffer.get_data(
                self.args.minibatch_size, transform=self.transform)
            loss += self.loss(self.net(buf_inputs), buf_labels)
            # inputs = torch.cat((inputs, buf_inputs))
            # buf_id = torch.cat((task_id*torch.ones_like(labels),buf_id))
            # labels = torch.cat((labels, buf_labels))

        else:
            buf_id = task_id*torch.ones_like(labels)

        # outputs = self.net(inputs)
        # loss = self.take_multitask_loss(buf_id, outputs, labels, task_id)
        loss.backward()
        self.opt.step()
        self.opt.zero_grad()

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
            # inputs = torch.cat((inputs, buf_inputs))
            # buf_id = torch.cat((task_id*torch.ones_like(labels),buf_id))
            # labels = torch.cat((labels, buf_labels))

        else:
            buf_id = task_id*torch.ones_like(labels)

        # outputs = self.net(inputs)
        # loss = self.take_multitask_loss(buf_id, outputs, labels, task_id)
        loss.backward()
        self.opt.step()
        self.opt.zero_grad()
        return loss.item()



'''
import torch
from utils.buffer import Buffer
from utils.args import *
from models.utils.continual_model import ContinualModel
import numpy as np
import random
import os
np.random.seed(0)

def get_parser() -> ArgumentParser:
    parser = ArgumentParser(description='Continual learning via'
                                        ' Experience Replay.')
    add_management_args(parser)
    add_experiment_args(parser)
    add_rehearsal_args(parser)
    return parser


class Er(ContinualModel):
    NAME = 'er'
    COMPATIBILITY = ['class-il', 'domain-il', 'task-il', 'general-continual']

    def __init__(self, backbone, loss, args, transform):
        super(Er, self).__init__(backbone, loss, args, transform)
        self.buffer = Buffer(self.args.buffer_size, self.device)
        

    def begin_task(self, dataset):
        self.N_CLASSES_PER_TASK = dataset.N_CLASSES_PER_TASK
        path = os.path.join(os.getcwd(), 'resnet18.pth')
        if os.path.exists(path):
            self.net.load_state_dict(torch.load(path))
   
    # def end_task(self, dataset):
    #     task_id = dataset.i//dataset.N_CLASSES_PER_TASK  
    #     if task_id != 5:
    #         print('task_id', task_id)
    #         self.net.fc.weight.data[dataset.i:dataset.i+2] = self.net.fc.weight.data[dataset.i-2:dataset.i]
    #         self.net.fc.bias.data[dataset.i:dataset.i+2] = self.net.fc.bias.data[dataset.i-2:dataset.i]
 
    def end_task(self, dataset):
        dataset.task_id = dataset.i//dataset.N_CLASSES_PER_TASK 
        # print('task_id', dataset.task_id)
        # Other classifier Reuse
        task_id = dataset.i//dataset.N_CLASSES_PER_TASK 
        torch.save(self.net.state_dict(), 'resnet18.pth') 
        # temp = task_id
        # print('temp',temp)
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
            
    def take_multitask_loss0(self, logits, y, t):
        loss = 0.0
        # print('t',t)
        # print('logits', logits.shape)
        # print('y', y.unique())
        for i, _ in enumerate(y):
            offset1, offset2 = self.compute_offsets(t)
            offset2 = 200
            # print('off1', offset1)
            # print('yi', y[i])
            loss += self.loss(logits[i, offset1:offset2].unsqueeze(0), y[i].unsqueeze(0)-offset1)
        return loss/len(y)

    def take_multitask_loss(self, bt, logits, y, task_id):
        loss = 0.0
        for i, ti in enumerate(bt):
            if i < 32:
                offset1, offset2 = self.compute_offsets(ti)
                offset2 = 200
                # loss += self.loss(logits[i, offset1:offset2].unsqueeze(0), y[i].unsqueeze(0)-offset1)
            else:
                _, offset2 = self.compute_offsets(task_id)
                offset1 = 0
            loss += self.loss(logits[i, offset1:offset2].unsqueeze(0), y[i].unsqueeze(0)-offset1)
        return loss/len(bt)

    def observe(self, inputs, labels, not_aug_inputs, task_id):
        # M = 10
        real_batch_size = inputs.shape[0]
        # real_batch_size = int(np.random.randint(1, min(M, real_batch_size), size=1))
        perm = torch.randperm(real_batch_size)
        # print('perm', perm.shape)
        # print('labels',labels.shape)
        inputs, labels = inputs[perm], torch.tensor(labels[perm],dtype=torch.long)
        not_aug_inputs = not_aug_inputs[perm]
        # self.opt.zero_grad()
        # outputs = self.net(inputs)
        # loss = self.loss(outputs, labels)
        # loss.backward()
        # self.opt.step()       
        # grad = torch.autograd.grad(loss, self.net.weights)
        # print('grad_norm', torch.norm(grad))
        # loss = self.take_multitask_loss0(outputs, labels, task_id)
        if not self.buffer.is_empty():
            buf_inputs, buf_labels, buf_id = self.buffer.get_data(
                self.args.minibatch_size, transform=self.transform)
            inputs = torch.cat((inputs, buf_inputs))
            buf_id = torch.cat((task_id*torch.ones_like(labels),buf_id))
            labels = torch.cat((labels, buf_labels))
            # outputs = self.net(buf_inputs)
            # loss_buf = self.loss(outputs, buf_labels)
            # self.opt.zero_grad()
            # loss_buf.backward()
            # self.opt.step()

            # grad_buf = torch.autograd.grad(loss_buf, self.net.weights)
            # print('gradbuf_norm', torch.norm(grad_buf))
            # loss += loss_buf
            # loss += self.take_multitask_loss(buf_id, outputs, buf_labels, task_id)
            # buf_outputs = self.net(buf_inputs)
            # loss += self.loss(buf_outputs, buf_labels)

        else:
            buf_id = task_id*torch.ones_like(labels)
        outputs = self.net(inputs)
        # outputs = self.net(inputs, mode = 'surrogate')
        # pseudo_labels = np.arange(task_id*2, task_id*2+2)
        # pseudo_labels = random.choices(pseudo_labels,k=len(outputs))
        # pseudo_labels = torch.tensor(pseudo_labels).cuda()
        # print('outputs',outputs)
        loss = self.take_multitask_loss(buf_id, outputs, labels, task_id)
        # loss = self.loss(outputs, labels)
        # if task_id !=4:
        #     loss += 0 * self.loss(outputs,pseudo_labels)
        # print('loss',loss)
        loss.backward()
        self.opt.step()
        self.opt.zero_grad()

        # update the balance classifier
        # self.opt1.zero_grad()
        # if not self.buffer.is_empty():
        #     buf_inputs, buf_labels, buf_id = self.buffer.get_data(
        #         50, transform=self.transform)           
        #     outputs = self.net(buf_inputs)
        #     # buf_labels = F.one_hot(buf_labels, num_classes=  self.Total_classes)
        #     # smoothing = 0.1
        #     # buf_labels = (1-smoothing) *buf_labels + smoothing/(self.Total_classes)
        #     # loss = F.binary_cross_entropy_with_logits(outputs, buf_labels)
        #     loss = self.loss(outputs,buf_labels)
        #     # print('b_loss', loss)
        #     loss.backward()
        #     self.opt1.step()
        #     self.opt1.zero_grad()

        self.buffer.add_data(examples=not_aug_inputs,
                            labels=labels,
                            task_labels=task_id*torch.ones_like(labels))

        return loss.item()

    def observe_test(self, inputs, labels, task_id):
        # print('ob_test', labels.unique())
        real_batch_size = inputs.shape[0]
        perm = torch.randperm(real_batch_size)
        inputs, labels = inputs[perm], torch.tensor(labels[perm],dtype=torch.long)
        # self.opt.zero_grad()
        # outputs = self.net(inputs)
        # loss = self.loss(outputs, labels)

        if not self.buffer.is_empty():
            buf_inputs, buf_labels, buf_id = self.buffer.get_data(
                self.args.minibatch_size, transform=self.transform)
            inputs = torch.cat((inputs, buf_inputs))
            buf_id = torch.cat((task_id*torch.ones_like(labels),buf_id))
            labels = torch.cat((labels, buf_labels))
            # outputs = self.net(buf_inputs)
            # loss_buf = self.loss(outputs, buf_labels)
            # self.opt.zero_grad()
            # loss_buf.backward()
            # self.opt.step()

            # grad_buf = torch.autograd.grad(loss_buf, self.net.weights)
            # print('gradbuf_norm', torch.norm(grad_buf))
            # loss += loss_buf
        else:
            buf_id = task_id*torch.ones_like(labels)
        outputs = self.net(inputs)
        # outputs = self.net(inputs, mode = 'surrogate')
        # loss = self.loss(outputs, labels)
        loss = self.take_multitask_loss(buf_id, outputs, labels, task_id)
        # loss = self.take_multitask_loss0(outputs, labels, task_id)
        # torch.save(fea, 'er_fea.pt')

        loss.backward()
        self.opt.step()

        # update the balance classifier
        # self.opt1.zero_grad()
        # if not self.buffer.is_empty():
        #     buf_inputs, buf_labels, buf_id = self.buffer.get_data(
        #         32, transform=self.transform)           
        #     outputs = self.net(buf_inputs)
        #     # buf_labels = F.one_hot(buf_labels, num_classes=  self.Total_classes)
        #     # smoothing = 0.1
        #     # buf_labels = (1-smoothing) *buf_labels + smoothing/(self.Total_classes)
        #     # loss = F.binary_cross_entropy_with_logits(outputs, buf_labels)
        #     loss = self.loss(outputs,buf_labels)
        #     # print('b_loss', loss)
        #     loss.backward()
        #     self.opt1.step()
        #     self.opt1.zero_grad()
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

# # Copyright 2020-present, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Davide Abati, Simone Calderara.
# # All rights reserved.
# # This source code is licensed under the license found in the
# # LICENSE file in the root directory of this source tree.

# # Copyright 2020-present, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Davide Abati, Simone Calderara.
# # All rights reserved.
# # This source code is licensed under the license found in the
# # LICENSE file in the root directory of this source tree.

# from functools import reduce
# import torch
# from utils.buffer import Buffer
# from utils.ring_buffer import RingBuffer

# from utils.args import *
# from models.utils.continual_model import ContinualModel
# from collections import OrderedDict
# import torch.nn.functional as F
# import numpy as np
# np.random.seed(0)
# def get_parser() -> ArgumentParser:
#     parser = ArgumentParser(description='Continual learning via'
#                                         ' Experience Replay.')
#     add_management_args(parser)
#     add_experiment_args(parser)
#     add_rehearsal_args(parser)
#     parser.add_argument('--grad_clip_norm', type=float, help='learning rate', default=1.0)
#     return parser

# class Er(ContinualModel):
#     NAME = 'er'
#     COMPATIBILITY = ['class-il', 'domain-il', 'task-il', 'general-continual']

#     def __init__(self, backbone, loss, args, transform):
#         super(Er, self).__init__(backbone, loss, args, transform)
#         self.buffer = Buffer(self.args.buffer_size, self.device)
#         # self.buffer = RingBuffer(self.args.buffer_size, self.device, 5)
#         # self.fast_weight = self.net.get_fast_weight()

#     def begin_task(self, dataset):
#         self.N_CLASSES_PER_TASK = dataset.N_CLASSES_PER_TASK

#     def compute_offsets(self, task):
#         # mapping from classes [1-100] to their idx within a task
#         offset1 = task * self.N_CLASSES_PER_TASK 
#         offset2 = (task + 1) * self.N_CLASSES_PER_TASK 
#         return int(offset1), int(offset2)       
            
#     def take_multitask_loss(self, bt, logits, y, task_id):
#         loss = 0.0
#         for i, ti in enumerate(bt):
#             # if ti == task_id:
#             if i < 10:
#                 offset1, offset2 = self.compute_offsets(ti)
#                 # offset1 = 0
#             else:
#                 _, offset2 = self.compute_offsets(task_id)
#                 offset1 = 0
#             # print('offset1',offset1)
#             loss += self.loss(logits[i, offset1:offset2].unsqueeze(0), y[i].unsqueeze(0)-offset1)
#             # lam = torch.exp(torch.tensor(-ti/5)) # for cifar10
#             # loss += lam*self.loss(logits[i, offset1:offset2].unsqueeze(0), y[i].unsqueeze(0)-offset1)
#             # loss += self.loss(logits[i, offset1:offset2].unsqueeze(0), y[i].unsqueeze(0)-offset1)
#             # offset1, offset2 = self.compute_offsets(task_id)
#             # offset1 = 0
#             # loss += self.loss(logits[i, offset1:offset2].unsqueeze(0), y[i].unsqueeze(0)-offset1)
#         return loss/len(bt)

#     # def take_multitask_loss(self, bt, logits, y):
#     #     loss = 0.0
#     #     for i, ti in enumerate(bt):
#     #         offset1, offset2 = self.compute_offsets(ti)
#     #         loss += self.loss(logits[i, offset1:offset2].unsqueeze(0), y[i].unsqueeze(0)-offset1)
#     #     return loss/len(bt)

#     def virtual_update(self, x, y, t):
#         self.fast_weight = self.net.get_fast_weight()
#         offset1, offset2 = self.compute_offsets(t)
#         logits = self.net.functional_forward(x, fast_weight= self.fast_weight)
#         logits = logits[:, offset1:offset2]
#         loss = self.loss(logits, y-offset1)
#         # print('loss',loss)
#         # print('net', self.net.get_fast_weight().values)
#         grad = list(torch.autograd.grad(loss, self.fast_weight.values()))
        
#         if self.args.grad_clip_norm:
#             for i in range(len(grad)):
#                 grad[i] = torch.clamp(grad[i], min=-self.args.grad_clip_norm, max=self.args.grad_clip_norm)
#         grad = tuple(grad)
#         fast_weight = OrderedDict(
#                     (name, p - self.args.lr * g)  for (name, p), g in zip(self.net.named_parameters(), grad)
#                 )
#         return fast_weight
        
#     def MIR_select_buffer(self,fast_weight, task_id):
#         buf_inputs, buf_labels, buf_id  = self.buffer.get_all_saved_data(task_id)
#         with torch.no_grad():
#             logits_track_pre = self.net.functional_forward(buf_inputs, fast_weight= self.net.get_fast_weight())
#             # print('logits_track_pre',logits_track_pre.shape)
#             logits_track_pre_stack = []
#             buf_labels_pre_stack = []
#             for i in range(logits_track_pre.shape[0]):
#                 offset1, offset2 = self.compute_offsets(buf_id[i])
#                 logits_track_pre_stack.append(logits_track_pre[i,offset1:offset2])
#                 buf_labels_pre_stack.append(buf_labels[i]-offset1)
#                 # buf_labels[i] = buf_labels[i] - offset1
#             logits_track_pre_stack= torch.stack(logits_track_pre_stack)
#             buf_labels_pre_stack = torch.stack(buf_labels_pre_stack )
#             pre_loss = F.cross_entropy(logits_track_pre_stack, buf_labels_pre_stack, reduction='none')
            
#             logits_track_post = self.net.functional_forward(buf_inputs, fast_weight= fast_weight)
#             logits_track_post_stack = []
#             buf_labels_post_stack = []
#             for i in range(logits_track_post.shape[0]):
#                 offset1, offset2 = self.compute_offsets(buf_id[i])
#                 logits_track_post_stack.append(logits_track_post[i,offset1:offset2])
#                 buf_labels_post_stack.append(buf_labels[i]-offset1) 
#                 # buf_labels[i] = buf_labels[i] # have already abstract the offset1 in pre calculation
#             logits_track_post_stack = torch.stack(logits_track_post_stack )
#             buf_labels_post_stack = torch.stack(buf_labels_post_stack)
#             # print('logits_track_post',logits_track_post_stack.shape)
#             post_loss = F.cross_entropy(logits_track_post_stack, buf_labels_post_stack, reduction='none')        
#             scores = post_loss - pre_loss 
#             # print('scores', scores.shape)
#             big_ind = scores.sort(descending=True)[1][:self.args.minibatch_size]
#         return buf_inputs[big_ind],buf_labels[big_ind],buf_id[big_ind]
            
#     def observe(self, inputs, labels, not_aug_inputs, task_id):
#         # print('task_id', task_id)
#         # M = 10
#         real_batch_size = inputs.shape[0]
#         # real_batch_size = int(np.random.randint(1, min(M, real_batch_size), size=1))
#         perm = torch.randperm(real_batch_size)
#         inputs, labels = inputs[perm], torch.tensor(labels[perm],dtype=torch.long)
#         not_aug_inputs = not_aug_inputs[perm]

#         self.opt.zero_grad()
#         if not self.buffer.is_empty():
#             # if task_id == 0:
#             if task_id <= 10:
#             # if task_id == 0:
#                 buf_inputs, buf_labels, buf_id = self.buffer.get_data(
#                     self.args.minibatch_size, transform=self.transform)               
#                 inputs = torch.cat((inputs, buf_inputs))
#                 buf_id = torch.cat((task_id*torch.ones_like(labels),buf_id))
#                 labels = torch.cat((labels, buf_labels))
#             else:
#                 fast_weight = self.virtual_update(inputs, labels, task_id)
#                 buf_inputs, buf_labels, buf_id = self.MIR_select_buffer(fast_weight, task_id)
#                 buf_id = task_id*torch.ones_like(labels)
#                 inputs = torch.cat((inputs, buf_inputs))
#                 buf_id = torch.cat((task_id*torch.ones_like(labels),buf_id))
#                 labels = torch.cat((labels, buf_labels))
            
#             # if task_id != 0:
#             #     buf_inputs, buf_labels, buf_id = self.MIR_select_buffer(fast_weight, task_id)
#             #     # buf_inputs, buf_labels, buf_id = self.buffer.get_data(
#             #     #     self.args.minibatch_size, transform=self.transform)
#             #     print('buf_labels', buf_labels)
#             #     inputs = torch.cat((inputs, buf_inputs))
#             #     buf_id = torch.cat((task_id*torch.ones_like(labels),buf_id))
#             #     labels = torch.cat((labels, buf_labels))
#             # else:
#             #     buf_id = task_id*torch.ones_like(labels)
#         else:
#             buf_id = task_id*torch.ones_like(labels)
#         # buf_id = task_id*torch.ones_like(labels)
#         outputs = self.net(inputs)
#         loss = self.loss(outputs, labels)
#         # loss = self.take_multitask_loss(buf_id, outputs, labels, task_id)
#         # loss = self.take_multitask_loss(buf_id, outputs, labels)

#         # loss = self.loss(outputs, labels)
#         loss.backward()
#         self.opt.step()

#         self.buffer.add_data(examples=not_aug_inputs,
#                              labels=labels,#[:real_batch_size],
#                              task_labels=task_id*torch.ones_like(labels))

#         return loss.item()

'''