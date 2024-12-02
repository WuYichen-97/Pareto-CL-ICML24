# Copyright 2022-present, Lorenzo Bonicelli, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import numpy # needed (don't change it)
import importlib
import os
import sys
import socket
mammoth_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(mammoth_path)
sys.path.append(mammoth_path + '/datasets')
sys.path.append(mammoth_path + '/backbone')
sys.path.append(mammoth_path + '/models')

from datasets import NAMES as DATASET_NAMES
from models import get_all_models
from argparse import ArgumentParser
from utils.args import add_management_args
from datasets import ContinualDataset
from utils.continual_training import train as ctrain
from datasets import get_dataset
from models import get_model
from utils.training import train#, evaluate0
from utils.best_args import best_args
from utils.conf import set_random_seed
import setproctitle
import torch
import uuid
import datetime
import wandb

import time
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def parse_args(): #
    parser = ArgumentParser(description='mammoth', allow_abbrev=False)
    # choose the model want to run
    parser.add_argument('--model', type=str, required=True,
                        help='Model name.', choices=get_all_models()) # duplicate introduce in add_experiment_args 注释掉了args里面的model
    torch.set_num_threads(4)
    # add_management_args(parser)
    args = parser.parse_known_args()[0]  # parser.parse_args()[0] (运行报错) 这样在sh文件里面加入其他命令行的时候, 这里还没添加，会报错. 所以必须用parse_known_args()
    mod = importlib.import_module('models.' + args.model)
    # save the best parameters in the best_args.py file
    # if args.load_best_args:
    #     add_management_args(parser)
    #     args = parser.parse_known_args()[0]
    #     parser.add_argument('--dataset', type=str, required=True,
    #                         choices=DATASET_NAMES,
    #                         help='Which dataset to perform experiments on.')
    #     if hasattr(mod, 'Buffer'):
    #         parser.add_argument('--buffer_size', type=int, required=True,
    #                             help='The size of the memory buffer.')
    #     args = parser.parse_args()
    #     if args.model == 'joint':
    #         best = best_args[args.dataset]['sgd']
    #     else:
    #         best = best_args[args.dataset][args.model]
    #     if hasattr(mod, 'Buffer'):
    #         best = best[args.buffer_size]
    #     else:
    #         best = best[-1]
    #     get_parser = getattr(mod, 'get_parser')
    #     parser = get_parser()
    #     to_parse = sys.argv[1:] + ['--' + k + '=' + str(v) for k, v in best.items()]
    #     to_parse.remove('--load_best_args')
    #     args = parser.parse_args(to_parse)
    #     if args.model == 'joint' and args.dataset == 'mnist-360':
    #         args.model = 'joint_gcl'
    # else:
    #     get_parser = getattr(mod, 'get_parser')
    #     parser = get_parser()
    #     args = parser.parse_args()
    get_parser = getattr(mod, 'get_parser')
    parser = get_parser()
    # other configurations are shown in args.py
    args = parser.parse_args()

    if args.seed is not None:
        set_random_seed(args.seed)

    return args

def main(args=None):
    # lecun_fix()
    # print('arg',args.model)
    if args is None:
        args = parse_args()
    # wandb.init(project = 'Cifar10-resnet', name=args.model+'w_mask')
    # wandb.init(project = 'Cifar10-mask', name=args.model)
    dataset = get_dataset(args)

    backbone = dataset.get_backbone() # 在 datasets.perm_mnist.py
    loss = dataset.get_loss()
    model = get_model(args, backbone, loss, dataset.get_transform())

    # print parameters
    print("Total trainable parameters: ", count_parameters(model))

    # wandb.watch(model,log='all')
    if isinstance(dataset, ContinualDataset):
        # print('!!!!!!!!!!!!!!!!!!!!!!!')
        # _, test_loader =  dataset.get_data_loaders(True)
        # accs = evaluate0(model,dataset)
        # print('\nAcc list:', accs[0])
        # print('\nTask list:', accs[1])
        start_time = time.time()
        # print('start time is:', start_time)
        train(model, dataset, args)
        end_time = time.time()
        total_time = end_time - start_time
        print('running time is: {:.2f}s'.format(total_time)) 
    else:
        assert not hasattr(model, 'end_task') or model.NAME == 'joint_gcl'
        ctrain(args)


if __name__ == '__main__':
    main()
