import quadprog
import numpy as np
import torch
from models.utils.continual_model import ContinualModel
from utils.buffer import Buffer
from utils.args import *
from utils.pareto import MinNormSolver

from collections import OrderedDict
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F

from collections import Counter
# from opt_einsum import contract

def get_parser() -> ArgumentParser:
    parser = ArgumentParser(description='Continual learning via'
                                        ' Gradient Episodic Memory.')
    add_management_args(parser)
    add_experiment_args(parser)
    add_rehearsal_args(parser)
    # remove minibatch_size from parser
    for i in range(len(parser._actions)):
        if parser._actions[i].dest == 'minibatch_size':
            del parser._actions[i]
            break

    parser.add_argument('--gamma', type=float, default=None,
                        help='Margin parameter for Pareto-cl.')
    return parser


def store_grad(params, grads, grad_dims):
    """
        This stores parameter gradients of past tasks.
        pp: parameters
        grads: gradients
        grad_dims: list with number of parameters per layers
    """
    # store the gradients
    grads.fill_(0.0)
    count = 0
    for param in params():
        if param.grad is not None:
            begin = 0 if count == 0 else sum(grad_dims[:count])
            end = np.sum(grad_dims[:count + 1])
            
            # grads[begin: end].copy_(param.grad.data.view(-1))

            grads_clamp = torch.clamp(param.grad.data, min= -10, max= 10)
            grads[begin: end].copy_(grads_clamp.data.view(-1))
        count += 1


def overwrite_grad(params, newgrad, grad_dims):
    """
        This is used to overwrite the gradients with a new gradient
        vector, whenever violations occur.
        pp: parameters
        newgrad: corrected gradient
        grad_dims: list storing number of parameters at each layer
    """
    count = 0
    for param in params():
        if param.grad is not None:
            begin = 0 if count == 0 else sum(grad_dims[:count])
            end = sum(grad_dims[:count + 1])
            this_grad = newgrad[begin: end].contiguous().view(
                param.grad.data.size())
            param.grad.data.copy_(this_grad)
        count += 1

def project2cone2(gradient, memories, margin=0.5, eps=1e-3):
    """
        Solves the GEM dual QP described in the paper given a proposed
        gradient "gradient", and a memory of task gradients "memories".
        Overwrites "gradient" with the final projected update.

        input:  gradient, p-vector
        input:  memories, (t * p)-vector
        output: x, p-vector
    """
    memories_np = memories.cpu().t().double().numpy()
    gradient_np = gradient.cpu().contiguous().view(-1).double().numpy()
    n_rows = memories_np.shape[0]
    self_prod = np.dot(memories_np, memories_np.transpose())
    self_prod = 0.5 * (self_prod + self_prod.transpose()) + np.eye(n_rows) * eps
    grad_prod = np.dot(memories_np, gradient_np) * -1
    G = np.eye(n_rows)
    h = np.zeros(n_rows) + margin
    v = quadprog.solve_qp(self_prod, grad_prod, G, h)[0]
    x = np.dot(v, memories_np) + gradient_np
    gradient.copy_(torch.from_numpy(x).view(-1, 1))

class paretocl(ContinualModel):
    NAME = 'paretocl'
    COMPATIBILITY = ['class-il', 'domain-il', 'task-il']

    def __init__(self, backbone, loss, args, transform):
        super(paretocl, self).__init__(backbone, loss, args, transform)
        self.current_task = 0
        self.current_task0 =-1
        self.buffer = Buffer(self.args.buffer_size, self.device)

        # Allocate temporary synaptic memory
        self.grad_dims = []
        for pp in self.parameters():
            self.grad_dims.append(pp.data.numel())
        # print('dims', self.grad_dims)

        self.grads_cs = []
        self.grads_cs.append(torch.zeros(
            np.sum(self.grad_dims)).to(self.device))
        self.grads_da = torch.zeros(np.sum(self.grad_dims)).to(self.device)
        self.alpha = 0
        self.alpha = torch.tensor([ ]).cuda()
        

    def begin_task(self, dataset):
        self.N_TASKS = dataset.N_TASKS
        self.N_CLASSES_PER_TASK = dataset.N_CLASSES_PER_TASK
        self.train_loader = dataset.train_loader
        self.fast_weight = self.net.get_fast_weight()
        self.grad_new = []
        self.task_id = -1
        self.step = 0
        self.cos_distance = []
        
    def end_task(self, dataset):
        self.current_task += 1
        self.grads_cs.append(torch.zeros(
            np.sum(self.grad_dims)).to(self.device))
        self.alpha = torch.cat((self.alpha, torch.tensor([0]).cuda()))
 
        
    def compute_offsets(self, task):
        # mapping from classes [1-100] to their idx within a task
        offset1 = task * self.N_CLASSES_PER_TASK 
        offset2 = (task + 1) * self.N_CLASSES_PER_TASK 
        return int(offset1), int(offset2)       
            

    def take_loss(self, t, logits, y):
        # compute loss on data from a single task
        offset1, offset2 = self.compute_offsets(t)
        loss = self.loss(logits[:, offset1:offset2], y-offset1)
        return loss

    def take_multitask_loss(self, bt, logits, y, task_id):
        loss = 0.0
        for i, ti in enumerate(bt):
            if i < 300:
                offset1, offset2 = self.compute_offsets(ti)
                loss += self.loss(logits[i].unsqueeze(0), y[i].unsqueeze(0))
            else:
                _, offset2 = self.compute_offsets(task_id)
                offset1 = 0
                loss += 0.5*self.loss(logits[i, offset1:offset2].unsqueeze(0), y[i].unsqueeze(0)-offset1)
        return loss/len(bt)

    def inner_loop(self, x, y, t, N):       
        fast_weights = self.net.get_fast_weight()
        logits = self.net.functional_forward(x, fast_weight= fast_weights)
        loss = self.take_loss(t,logits,y)
        grad = list(torch.autograd.grad(loss, fast_weights.values(), create_graph=True, retain_graph=True))
        for i in range(len(grad)):
            grad[i] = torch.clamp(grad[i], min= -2, max= 2)

        self.fast_weight = OrderedDict(
           (name, p - 0.03 * g)  for (name, p), g in zip(fast_weights.items(), grad )
        )  
        _, pred = torch.max(logits.data, 1)
        correct = torch.eq(pred, y).sum()
        return correct, logits, loss
    
    def consectuive_step_cosine(self, grad):
        if len(self.last_step_grad) ==0:
            grad_temp = grad.view(-1)
            self.last_step_grad = grad_temp
        else:
            grad_temp = grad.view(-1)
            sim = torch.dot(grad_temp, self.last_step_grad)
            sim = sim/(torch.norm(grad_temp)*torch.norm(self.last_step_grad))
            self.cos_distance.append(sim)
            self.last_step_grad = grad_temp
            

    def observe(self, inputs, labels, not_aug_inputs,t):
        real_batch_size = inputs.shape[0]
        perm = torch.randperm(real_batch_size)
        inputs, labels = inputs[perm], torch.tensor(labels[perm], dtype=torch.long)
        not_aug_inputs = not_aug_inputs[perm]
        weights = torch.ones(t+1).cuda()/(t+1)
        if self.task_id != t:
            self.task_id = t 
            self.weights = torch.ones(t).cuda()/(t)
        vecs = (torch.ones(t+1)/(1)).cuda()

        is_training = False
        if t==0:
            self.opt.zero_grad()
            outputs = self.forward(inputs)
            loss = self.loss(outputs, labels) 
            loss.backward()        
            self.opt.step()
            self.buffer.add_data(examples=not_aug_inputs,
                                labels=labels,
                                task_labels=t*torch.ones_like(labels))
        
        elif t>0 and self.step>5:
            if not self.buffer.is_empty():
                buf_inputs, buf_labels, buf_task_labels = self.buffer.get_data(
                    300, transform=self.transform)
                count = Counter(buf_task_labels.flatten().tolist())

                N = [(buf_task_labels == i).sum() for i in buf_task_labels.unique() ] 
                for tt in buf_task_labels.unique():
                    cur_task_inputs = buf_inputs[buf_task_labels == tt]#[: int(N[tt]/2)]
                    cur_task_labels = buf_labels[buf_task_labels == tt]#[: int(N[tt]/2)]
                    if tt ==  t:
                        cur_task_inputs = torch.cat((inputs, cur_task_inputs))
                        cur_task_labels = torch.cat((labels, cur_task_labels))
                        buf_task_labels1 = torch.cat((t*torch.ones_like(labels),buf_task_labels ))

                    correct_inner, logits, loss = self.inner_loop(cur_task_inputs, cur_task_labels, tt, len(buf_task_labels.unique()))
                    cur_task_inputs_qry = torch.cat((inputs, buf_inputs))
                    cur_task_labels_qry = torch.cat((labels, buf_labels))
                    buf_task_labels1 = torch.cat((t*torch.ones_like(labels), buf_task_labels ))

                    if tt ==0:
                        _ = self.net(cur_task_inputs_qry) # update the BN when the network is ResNet18
                    cur_task_outputs = self.net.functional_forward(cur_task_inputs_qry, fast_weight= self.fast_weight)
                    penalty = self.take_multitask_loss(buf_task_labels1, cur_task_outputs, cur_task_labels_qry, t) 
                    penalty.backward()
                    store_grad(self.parameters, self.grads_cs[tt], self.grad_dims) # self.grad_cs  is the hyper-gradient.
                
                if t ==1:
                    grads = (torch.stack(self.grads_cs[:-1])).mean(0) + self.grads_cs[-1] #(torch.stack(self.grads_cs)).mean(0)   
                else:
                    self.weights = MinNormSolver.find_min_norm_element_init(torch.stack(self.grads_cs[:-1]).clone(), self.weights.detach())
                    grads = (torch.stack(self.grads_cs[:-1])* self.weights.unsqueeze(1)).sum(0) + self.grads_cs[-1]

                overwrite_grad(self.parameters, grads.squeeze(), self.grad_dims)
                self.opt.step()
                self.opt.zero_grad()
                self.buffer.add_data(examples=not_aug_inputs,
                                    labels=labels,
                                    task_labels=t*torch.ones_like(labels))
        else:
            self.opt.zero_grad()
            outputs = self.forward(inputs)
            loss = self.loss(outputs, labels) 
            self.buffer.add_data(examples=not_aug_inputs,
                                labels=labels,
                                task_labels=t*torch.ones_like(labels))
        self.step +=1
        return loss.item()
