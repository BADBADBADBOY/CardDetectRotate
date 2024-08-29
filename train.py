#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Author : BADBADBADBOY
# @File : train.py
# @Time : 2024/8/20 21:22

import os
import time
import torch
from dataload.dataload import DataLoad
import random
import argparse
import numpy as np
import torch.backends.cudnn as cudnn
import torch.distributed as dist
from model.loss import CtdetLoss
import torch.optim as optim
from utils.utils import AverageMeter
from config import *
from optimizer import *

def init_seeds(seed=0, cuda_deterministic=True):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    # Speed-reproducibility tradeoff https://pytorch.org/docs/stable/notes/randomness.html
    if cuda_deterministic:  # slower, more reproducible
        cudnn.deterministic = True
        cudnn.benchmark = False
    else:  # faster, less reproducible
        cudnn.deterministic = False
        cudnn.benchmark = True

def trainProgram(opt):
    
    local_rank = opt.local_rank
    init_seeds(local_rank)
    
    dist.init_process_group(backend='nccl')
    torch.cuda.set_device(local_rank)
    
    assert BASE_MODEL in MODEL_SUPPORT,"not support this model !!!"

    if BASE_MODEL == "resnet":
        from model.models_resnet import CardDetectionCorrectionModel
        model = CardDetectionCorrectionModel(num_layers = NUM_LAYER, need_ftype=NEED_FTYPE)
    elif BASE_MODEL == "lcnet":
        from model.models_lcnet import CardDetectionCorrectionModel
        model = CardDetectionCorrectionModel(ratio = MODEL_RATIO, need_ftype=NEED_FTYPE)
    elif BASE_MODEL == "replcnet":
        from model.models_replcnet import CardDetectionCorrectionModel
        model = CardDetectionCorrectionModel(ratio=MODEL_RATIO, need_ftype=NEED_FTYPE)

    if OPTIM_METHOD == "Adam":
        optimizer = AdamDecay(OPTIM_CONFIG,model.parameters())
    else:
        optimizer = SGDDecay(OPTIM_CONFIG, model.parameters())

    criterion = CtdetLoss(HM_WEIGHT,WH_WEIGHT,OFFSET_WEIGHT,ANGLE_WEIGHT)

    if LOAD_PRE_MODEL:
        if LOAD_TYPE == "resnet18":
            pre_dict = torch.load(PRE_MODEL_PATH)
            new_pre_dict = {}
            for key in pre_dict.keys():
                if "layer4" in key:
                    new_pre_dict[key] = model.state_dict()[key]
                else:
                    new_pre_dict[key] = pre_dict[key]
            model.load_state_dict(new_pre_dict,strict=False)

        elif LOAD_TYPE == "modelscope" and BASE_MODEL=="resnet":
            model.load_state_dict(torch.load(PRE_MODEL_PATH)['state_dict'])
        else:
            model.load_state_dict(torch.load(PRE_MODEL_PATH))
            
    train_dataset = DataLoad(img_dir,gt_dir,MAX_OBJS, INPUT_H, INPUT_W)
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
            
    train_loader = torch.utils.data.DataLoader(
          train_dataset,
          batch_size=BATCH_SIZE,
          sampler=train_sampler,
          num_workers=NUM_WORKER,
          pin_memory=True,
          drop_last=True
      )
    model = model.cuda(local_rank)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], find_unused_parameters=True)
    criterion = criterion.cuda(local_rank)

    if dist.get_rank() == 0:
        print(model)
        if not os.path.exists(SAVE_PATH):
            os.mkdir(SAVE_PATH)
        fid = open(os.path.join(SAVE_PATH, 'loss_{}.txt'.format(BASE_MODEL)),'w+', encoding='utf-8')

    total_iter = len(train_loader) * NUM_EPOCH
    OPTIM_CONFIG['optimizer']['n_epoch'] = total_iter

    for epoch in range(NUM_EPOCH):
        train_sampler.set_epoch(epoch)
        current_lr = optimizer.param_groups[0]['lr']
        epoch_ave_total = AverageMeter()
        epoch_ave_hm = AverageMeter()
        epoch_ave_wh = AverageMeter()
        epoch_ave_reg = AverageMeter()
        epoch_ave_cls = AverageMeter()
        data_time_record = AverageMeter()
        model_time_record = AverageMeter()
        data_s = time.time()
        for iter_id, batch in enumerate(train_loader):
            data_e = time.time()
            train_iter = (len(train_loader) * epoch + iter_id)
            
            for key in batch.keys():
                batch[key] = batch[key].cuda(local_rank)

            model_s = time.time()
            outputs = model(batch['input'])
            loss, loss_stats = criterion(outputs, batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            model_e = time.time()
            model_time_record.update(model_e - model_s)
            data_time_record.update(data_e - data_s)

            epoch_ave_total.update(loss_stats['loss'].item())
            epoch_ave_hm.update(loss_stats['hm_loss'].item())
            epoch_ave_wh.update(loss_stats['wh_loss'].item())
            epoch_ave_reg.update(loss_stats['off_loss'].item())
            epoch_ave_cls.update(loss_stats['cls_loss'].item())
            if train_iter % SHOW_STEP == 0 and dist.get_rank() == 0:
                print_log = "Train iters:{}/{}\t".format(train_iter,total_iter)
                for key in loss_stats.keys():
                    print_log += "{}:{}".format(key,"%.4f"%loss_stats[key].item())
                    print_log += '\t'
                print_log += "data_time:{}".format("%.3f"%data_time_record.avg)
                print_log += "\tmodel_time:{}".format("%.3f"%model_time_record.avg)
                print_log += "\tLr:{}".format("%.8f"%current_lr)
                print(print_log)
            if train_iter % SAVE_ITERS == 0 and train_iter != 0 and dist.get_rank() == 0:
                torch.save(model.state_dict(),"{}/model_{}.pth".format(SAVE_PATH,train_iter))
                print("**************************************************************************")
                print("Save Iter {}/{}\t loss:{}\thm_loss:{}\twh_loss:{}\toff_loss:{}\tcls_loss:{}".format(train_iter,total_iter,"%.4f"%epoch_ave_total.avg,"%.4f"%epoch_ave_hm.avg,"%.4f"%epoch_ave_wh.avg,"%.4f"%epoch_ave_reg.avg,"%.4f"%epoch_ave_cls.avg))
                print("Save {} success!!".format("./model_save/model_{}.pth".format(train_iter)))
                print("**************************************************************************")
                fid.write("Save Iter {}/{}\t loss:{}\thm_loss:{}\twh_loss:{}\toff_loss:{}\tcls_loss:{}".format(train_iter,total_iter,"%.4f"%epoch_ave_total.avg,"%.4f"%epoch_ave_hm.avg,"%.4f"%epoch_ave_wh.avg,"%.4f"%epoch_ave_reg.avg,"%.4f"%epoch_ave_cls.avg)+'\n')
                fid.flush()
            adjust_learning_rate_poly(OPTIM_CONFIG, optimizer, train_iter)
            data_s = time.time()

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_rank', type=int, default=-1, help='DDP parameter, do not modify')
    opt = parser.parse_args()
    trainProgram(opt)