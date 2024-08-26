#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Author : BADBADBADBOY
# @File : dataload.py
# @Time : 2024/8/20 21:22

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import cv2
import os
import math
import torch
import numpy as np
import torch.utils.data as data
from utils.utils import order_points_new
from dataload.image_utils import flip, color_aug
from dataload.image_utils import get_affine_transform, affine_transform
from dataload.image_utils import gaussian_radius, draw_umich_gaussian, draw_msra_gaussian
from dataload.image_utils import draw_dense_reg
from dataload.image_utils import random_color_transform,random_brightness_adjust,random_white_mask,random_motion_blur,random_gaussian_blur


class DataLoad(data.Dataset):
    def __init__(self,img_dir, gt_dir, max_objs, input_h, input_w):
        img_file = os.listdir(img_dir)
        self.nSamples = len(img_file)
        self.file_list = []
        for file in img_file:
            self.file_list.append(file)
        self.img_dir = img_dir
        self.gt_dir = gt_dir
        self.max_objs = max_objs
        self.input_h, self.input_w = input_h, input_w
        self.down_ratio = 4
        self.mean = np.array([0.408, 0.447, 0.470],
                        dtype=np.float32).reshape(1, 1, 3)
        self.std = np.array([0.289, 0.274, 0.278],
                       dtype=np.float32).reshape(1, 1, 3)

    def load_gt(self,gt_file):
        gt_list = []
        with open(gt_file,'r',encoding='utf-8') as fid:
            for line in fid.readlines():
                line = line.split(',')
                line = [int(item) for item in line]
                gt_list.append(np.array(line))
        return gt_list
    
    def __len__(self):
        return self.nSamples
    
    def random_aug_img(self,img, ratio = 0.4):
        if np.random.rand() < ratio:
            img = random_color_transform(img)
        if np.random.rand() < ratio:
            img = random_brightness_adjust(img)
        if np.random.rand() < ratio:
            img = random_white_mask(img)
        if np.random.rand() < ratio:
            img = random_motion_blur(img)
        if np.random.rand() < ratio:
            img = random_gaussian_blur(img)
        return img
    
    def random_data(self,img,output_size=(512, 512)):
        h, w = img.shape[:2]
        center = np.array((w // 2, h // 2)) + np.array([np.random.randint(-64,64),np.random.randint(-64,64)])
        scale = max(h, w) * np.random.choice(np.arange(0.7, 1.3, 0.1))
        rot = np.random.randint(-10, 10)
        trans = get_affine_transform(center, scale=scale, rot=rot, output_size=output_size)
        img = cv2.warpAffine(img, trans, output_size)
        return img, trans
    
    def regular_data(self,img,output_size=(512, 512)):
        h, w = img.shape[:2]
        center = np.array((w // 2, h // 2)) 
        scale = max(h, w) 
        rot = 0
        trans = get_affine_transform(center, scale=scale, rot=rot, output_size=output_size)
        img = cv2.warpAffine(img, trans, output_size)
        return img, trans

    def __getitem__(self, index):
        img_path = os.path.join(self.img_dir,self.file_list[index])
        gt_path = os.path.join(self.gt_dir,self.file_list[index].split('.')[0] + '.txt')
        gt_objs = self.load_gt(gt_path)
        num_objs = min(len(gt_objs), self.max_objs)
        img = cv2.imread(img_path)
        img = self.random_aug_img(img, ratio = 0.4)
        
        if np.random.rand() > 0.5:
            inp, trans = self.random_data(img,(self.input_w, self.input_h))
        else:
            inp, trans = self.regular_data(img,(self.input_w, self.input_h))
        
        inp_show = cv2.resize(inp.copy(),None,fx=1/self.down_ratio,fy=1/self.down_ratio)
        inp = (inp.astype(np.float32) / 255.)
        inp = (inp - self.mean) / self.std
        inp = inp.transpose(2, 0, 1)
        output_h, output_w = self.input_h//self.down_ratio, self.input_w//self.down_ratio
        
        hm = np.zeros((1, output_h, output_w), dtype=np.float32)
        wh = np.zeros((self.max_objs, 8), dtype=np.float32)
        cls = np.zeros((4, output_h, output_w), dtype=np.float32)

        reg = np.zeros((self.max_objs, 2), dtype=np.float32)
        ind = np.zeros((self.max_objs), dtype=np.int64)
        reg_mask = np.zeros((self.max_objs), dtype=np.uint8)
        
        for k in range(num_objs):
            
            bbox = gt_objs[k][:4] ## (x,y,w,h) -> (x1,y1,x2,y2)
            poly = gt_objs[k][4:12].reshape(4,2)
            cl = gt_objs[k][12]
            
            
            poly[0] = affine_transform(poly[0], trans)
            poly[1] = affine_transform(poly[1], trans)
            poly[2] = affine_transform(poly[2], trans)
            poly[3] = affine_transform(poly[3], trans)

            bbox[0:2] = affine_transform(bbox[0:2], trans)
            bbox[2:] = affine_transform(bbox[2:], trans)

            poly = poly / self.down_ratio
            bbox = bbox / self.down_ratio
            
            poly = order_points_new(poly)  
            poly = poly.reshape(-1)
            
            bbox_center = np.array([(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2])
            h, w = bbox[3] - bbox[1], bbox[2] - bbox[0]

            if (h > 0 and w > 0 and
                bbox_center[0]>0 and
                bbox_center[1]>0 and
                bbox_center[0]<output_w and
                bbox_center[1]<output_h):
            
                inp_show = cv2.drawContours(inp_show,[poly.reshape(4,2).astype(np.int32)],-1,(0,0,255),2)
                inp_show = cv2.circle(inp_show, tuple(bbox_center.astype(np.int32).tolist()), 1, (0, 0, 255), 1)
                
                radius = gaussian_radius((math.ceil(h), math.ceil(w)))
                radius = max(0, int(radius))

                ct = bbox_center
                ct_int = ct.astype(np.int32)

                draw_umich_gaussian(hm[0], ct_int, radius)
                draw_umich_gaussian(cls[cl], ct_int, radius)

                wh[k] = ct[0]-poly[0], ct[1]-poly[1], ct[0]-poly[2], ct[1]-poly[3], ct[0]-poly[4], ct[1]-poly[5], ct[0]-poly[6], ct[1]-poly[7]

                ind[k] = ct_int[1] * output_w + ct_int[0]

                reg[k] = ct - ct_int

                reg_mask[k] = 1
                
            cv2.imwrite('inp_show.jpg', inp_show)

        ret = {'input': inp, 'hm': hm, 'cls': cls, 'reg_mask': reg_mask, 'ind': ind, 'wh': wh, 'reg': reg}

        return ret