# Copyright 2020-present, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Davide Abati, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import math
import torch
import torch.nn.functional as F
from utils.buffer import Buffer
from datasets import NAMES as DATASET_NAMES
# from utils.args import *
from argparse import ArgumentParser
from models import get_all_models
from models.utils.continual_model import ContinualModel
# from src.utils.training import mask_classes_in_k
from collections import OrderedDict
import numpy as np

def get_parser():
    parser = ArgumentParser(description='Continual learning via La-MAML')
    parser.add_argument('--model', type=str, required=True,
                        help='Model name.', choices=get_all_models())
    parser.add_argument('--seed', type=int, required=True, help='seed')

    parser.add_argument('--dataset', type=str, required=True, choices=DATASET_NAMES,
                        help='Which dataset to perform experiments on.')
    parser.add_argument('--lr', type=float, required=True, help='Learning rate.')
    #parser.add_argument('--optim_wd', type=float, default=0., help='optimizer weight decay.')
    #parser.add_argument('--optim_mom', type=float, default=0., help='optimizer momentum.')
    #parser.add_argument('--optim_nesterov', type=int, default=0, help='optimizer nesterov momentum.')

    # train
    parser.add_argument('--n_epochs', type=int, help='Batch size.')
    parser.add_argument('--batch_size', type=int, help='Batch size.')
    parser.add_argument('--replay_batch_size', type=int, help='Batch size.')
    parser.add_argument('--buffer_size', type=int, required=True, help='The size of the memory buffer.')
    parser.add_argument('--minibatch_size', type=int, help='The batch size of the memory buffer.')

    # optimizer
    parser.add_argument('--optim_wd', type=float, default= 0, help='Weight_decay')
    parser.add_argument('--optim_mom', type=float, default=0.9, help='Weight_decay') # for cifar100 dataset
    # parser.add_argument('--optim_lr', type=float, help='learning rate')
    parser.add_argument('--grad_clip_norm', type=float, help='learning rate', default=2.0)
    #parser.add_argument('beta1',type=float)#, default = 0.9)
    #parser.add_argument('beta2',type=float)#, default = 0.999)

    # meta
    parser.add_argument('--alpha_initial', type=float, help='inner_loop learning rate', default = 0.15)
    parser.add_argument('--second_order', default=False, action='store_true')
    parser.add_argument('--asyn_update', default=False, action='store_true')
    parser.add_argument('--inner_batch_size', type=int, help='inner loop update using minibatch', default = 1)
    parser.add_argument('--meta_update_per_batch', type=int, default = 5)
    # other
    parser.add_argument('--csv_log', action='store_true', help='Enable csv logging')
    parser.add_argument('--tensorboard', action='store_true', help='Enable tensorboard logging')
    parser.add_argument('--validation', action='store_true', help='Test on the validation set')
    return parser



class Foml(ContinualModel):
    NAME = 'foml'
    COMPATIBILITY = ['class-il', 'domain-il', 'task-il', 'general-continual']

    def __init__(self, backbone, loss, args, transform):
        super(Foml, self).__init__(backbone, loss, args, transform)
        self.buffer = Buffer(self.args.buffer_size, self.device)

        self.task_lr = OrderedDict(
            [ (name+'lr',torch.nn.Parameter(
            self.args.alpha_initial * torch.ones_like(p, requires_grad=True, device=self.device))) for name,p in
                        self.net.named_parameters()]
        )
        self.opt_lr = torch.optim.SGD(self.task_lr.values(), lr=self.args.lr)
        self.meta_loss = None
        self.fast_weight = None
        self.observed_batch = None
        self.max_batch_in_minitask = None

    def begin_task(self, dataset):
        # clear meta-loss
        self.meta_loss = []
        self.reg_loss = []
        # clone the model
        self.fast_weight = self.net.get_fast_weight()
        # zero grad
        self.opt.zero_grad()
        self.opt_lr.zero_grad()
        self.observed_batch = 0
        self.max_batch_in_minitask = len(dataset.train_loader)

    def inner_loop(self, x: torch.Tensor, y: torch.Tensor) -> [torch.Tensor, torch.Tensor]:
        logits = self.net.functional_forward(x, fast_weight=self.fast_weight)
        loss = self.loss(logits, y) + 0.1 *self.l2_loss(self.fast_weight.items())
        grad = list(torch.autograd.grad(loss, self.fast_weight.values(), create_graph = self.args.second_order) )
        if self.args.grad_clip_norm:
            for i in range(len(grad)):
                grad[i] = torch.clamp(grad[i], min=-self.args.grad_clip_norm, max=self.args.grad_clip_norm)
        grad = tuple(grad)

        # for mnist
        # self.fast_weight = OrderedDict(
        #    (name, p - torch.nn.functional.relu(lr) * g)  for (name, p), g, lr in zip(self.fast_weight.items(), grad, self.task_lr.values())
        # )
        
        # for cifar10/100
        # self.fast_weight = OrderedDict(
        #    (name, p - lr * g)  for (name, p), g, lr in zip(self.fast_weight.items(), grad, self.task_lr.values())
        # )
        self.fast_weight = OrderedDict(
           (name, p - 0.1 * g)  for (name, p), g in zip(self.fast_weight.items(), grad)
        )


        _, pred = torch.max(logits.data, 1)
        correct = torch.eq(pred, y).sum()
        return correct

    def outer_loop(self):
        # update the parameters of backbone
        meta_loss = sum(self.meta_loss) / len(self.meta_loss)
        reg_loss = sum(self.reg_loss) / len(self.reg_loss)
        (meta_loss+0.1*reg_loss).backward()
        #meta_loss.backward()
        if self.args.grad_clip_norm:
            torch.nn.utils.clip_grad_norm_(self.net.parameters(), self.args.grad_clip_norm)
            torch.nn.utils.clip_grad_norm_(self.task_lr.values(), self.args.grad_clip_norm)

        # update the task_lr
        # self.opt_lr.step()
        if self.args.asyn_update:  #C-MAML：False/ La-MAML：True ??
            with torch.no_grad():
                for p, lr in zip(self.net.main.parameters(), self.task_lr.values()):
                    p.data = p.data - p.grad * 0.2
        else:
            self.opt.step()
        # zero grad and loss
        self.opt.zero_grad()
        self.opt_lr.zero_grad()
        # without regard this part
        # self.fast_weight = self.net.get_fast_weight()
        self.meta_loss = []
        self.reg_loss = []
        return meta_loss

    def l2_loss(self, a):
        a = [sum((p-p1)**2) for (name, p), (name1, p1) in zip(a, self.net.named_parameters())]
        diff = 0
        for i in range(len(a)):
             diff += sum(torch.flatten(a[i]))
        return  diff#.detach()

    def observe(self, inputs: torch.Tensor, labels: torch.Tensor, not_aug_inputs,t):
        correct = 0
        # M = 10
        inputs = torch.tensor(inputs, dtype=torch.float)
        labels = torch.tensor(labels, dtype=torch.long)
        real_batch_size = inputs.shape[0]
        # real_batch_size = int(np.random.randint(1, min(M, real_batch_size), size=1))
        num_inner_steps = math.ceil(real_batch_size / self.args.inner_batch_size)

        for i in range(self.args.meta_update_per_batch):
            perm = torch.randperm(real_batch_size)
            x_s, y_s = inputs[perm][:real_batch_size], labels[perm][:real_batch_size]
           

            if self.buffer.is_empty():
                x_q, y_q = x_s, y_s

            else:
                buf_inputs, buf_labels = self.buffer.get_data(
                    self.args.replay_batch_size, transform=self.transform)
                x_q = torch.cat((x_s, buf_inputs))
                y_q = torch.cat((torch.tensor(y_s, dtype=torch.long), buf_labels))

            # one step adaptation
            for j in range(num_inner_steps):
                x = x_s[j * self.args.inner_batch_size:(j + 1) * self.args.inner_batch_size]
                y = y_s[j * self.args.inner_batch_size:(j + 1) * self.args.inner_batch_size]
                y = torch.tensor(y, dtype=torch.long)
                correct_inner = self.inner_loop(x, y)
                # accumulate meta-loss for minibatchs:
                outer_logits = self.net.functional_forward(x_q, fast_weight=self.fast_weight)
                self.meta_loss.append(self.loss(outer_logits, y_q))
                self.reg_loss.append(self.l2_loss(self.fast_weight.items()))
                # record first meta update and first inner-loop
                if not j and not i:
                    correct += correct_inner
            # meta_update
            meta_loss = self.outer_loop()
        self.observed_batch += 1
        if self.observed_batch <= self.max_batch_in_minitask:
            self.buffer.add_data(examples=not_aug_inputs[:real_batch_size],
                                 labels=labels[:real_batch_size])
        return meta_loss.item()#, correct.item()

