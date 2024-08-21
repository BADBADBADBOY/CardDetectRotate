#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Author : BADBADBADBOY
# @File : optimizer.py
# @Time : 2024/8/20 22:07

import torch

def AdamDecay(config,parameters):
    optimizer = torch.optim.Adam(parameters, lr = config['optimizer']['base_lr'],
                                weight_decay = config['optimizer']['weight_decay'])
    return optimizer

def SGDDecay(config,parameters):
    optimizer = torch.optim.SGD(parameters, lr = config['optimizer']['base_lr'],
                               momentum = config['optimizer']['momentum'],
                               weight_decay = config['optimizer']['weight_decay'])
    return optimizer

def lr_poly(base_lr, epoch, max_epoch = 1200, factor = 0.9):
    return base_lr * (( 1 - float(epoch) / max_epoch) ** (factor))


def adjust_learning_rate_poly(config, optimizer, epoch):
    lr = lr_poly(config['optimizer']['base_lr'], epoch,
                 config['optimizer']['n_epoch'], config['optimizer']['factor'])
    optimizer.param_groups[0]['lr'] = lr