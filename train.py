#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Author : BADBADBADBOY
# @File : train.py
# @Time : 2024/8/20 21:22

import os
import time
import torch
from dataload.dataload import DataLoad
from model.models import CardDetectionCorrectionModel
from model.loss import CtdetLoss
import torch.optim as optim
from utils.utils import AverageMeter
from config import *
from optimizer import *

model = CardDetectionCorrectionModel(num_layers=18)

if OPTIM_METHOD == "Adam":
    optimizer = AdamDecay(OPTIM_CONFIG,model.parameters())
else:
    optimizer = SGDDecay(OPTIM_CONFIG, model.parameters())

criterion = CtdetLoss(HM_WEIGHT,WH_WEIGHT,OFFSET_WEIGHT,ANGLE_WEIGHT)

if torch.cuda.is_available():
    model = model.cuda()
    criterion = criterion.cuda()

if LOAD_TYPE == "resnet18":
    pre_dict = torch.load(PRE_MODEL_PATH)
    new_pre_dict = {}
    for key in pre_dict.keys():
        if "layer4" in key:
            new_pre_dict[key] = model.state_dict()[key]
        else:
            new_pre_dict[key] = pre_dict[key]
    model.load_state_dict(new_pre_dict,strict=False)
    
elif LOAD_TYPE == "modelscope":
    model.load_state_dict(torch.load(PRE_MODEL_PATH)['state_dict'])

train_loader = torch.utils.data.DataLoader(
      DataLoad(img_dir,gt_dir,MAX_OBJS, INPUT_H, INPUT_W),
      batch_size=BATCH_SIZE,
      shuffle=True,
      num_workers=NUM_WORKER,
      pin_memory=True,
      drop_last=True
  )

fid = open('./loss.txt','w+', encoding='utf-8')

if not os.path.exists(SAVE_PATH):
    os.mkdir(SAVE_PATH)

total_iter = len(train_loader) * NUM_EPOCH
OPTIM_CONFIG['optimizer']['n_epoch'] = total_iter

for epoch in range(NUM_EPOCH):
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
        
        if torch.cuda.is_available():
            for key in batch.keys():
                batch[key] = batch[key].cuda()
        
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
        if train_iter % SHOW_STEP == 0:
            print_log = "Train iters:{}/{}\t".format(train_iter,total_iter)
            for key in loss_stats.keys():
                print_log += "{}:{}".format(key,"%.4f"%loss_stats[key].item())
                print_log += '\t'
            print_log += "data_time:{}".format("%.3f"%data_time_record.avg)
            print_log += "\tmodel_time:{}".format("%.3f"%model_time_record.avg)
            print_log += "\tLr:{}".format("%.8f"%current_lr)
            print(print_log)
        if train_iter % SAVE_ITERS == 0 and train_iter != 0:
            torch.save(model.state_dict(),"{}/model_{}.pth".format(SAVE_PATH,train_iter))
            print("**************************************************************************")
            print("Save Iter {}/{}\t loss:{}\thm_loss:{}\twh_loss:{}\toff_loss:{}\tcls_loss:{}".format(train_iter,total_iter,"%.4f"%epoch_ave_total.avg,"%.4f"%epoch_ave_hm.avg,"%.4f"%epoch_ave_wh.avg,"%.4f"%epoch_ave_reg.avg,"%.4f"%epoch_ave_cls.avg))
            print("Save {} success!!".format("./model_save/model_{}.pth".format(train_iter)))
            print("**************************************************************************")
            fid.write("Save Iter {}/{}\t loss:{}\thm_loss:{}\twh_loss:{}\toff_loss:{}\tcls_loss:{}".format(train_iter,total_iter,"%.4f"%epoch_ave_total.avg,"%.4f"%epoch_ave_hm.avg,"%.4f"%epoch_ave_wh.avg,"%.4f"%epoch_ave_reg.avg,"%.4f"%epoch_ave_cls.avg)+'\n')
            fid.flush()
        adjust_learning_rate_poly(OPTIM_CONFIG, optimizer, train_iter)
        data_s = time.time()