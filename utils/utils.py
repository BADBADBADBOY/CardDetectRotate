#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Author : BADBADBADBOY
# @File : utils.py
# @Time : 2024/8/20 21:22

import copy
import math
import random

import cv2
import numpy as np
import torch
import torch.nn as nn
import importlib.metadata


def merge_images_horizontal(images, output_path="./show.jpg"):   
    # 确定目标高度（所有图像的目标高度）
    target_height = min(img.shape[0] for img in images)
    
    # 调整所有图像的大小，以使它们的高度一致
    resized_images = []
    for img in images:
        aspect_ratio = img.shape[1] / img.shape[0]  # 计算原始图像的宽高比
        new_width = int(target_height * aspect_ratio)  # 计算调整后的宽度
        resized_img = cv2.resize(img, (new_width, target_height))  # 调整图像大小
        resized_images.append(resized_img)
    
    # 计算合并后的总宽度
    total_width = sum(img.shape[1] for img in resized_images)
    
    # 创建一个新的空白图像
    merged_image = np.zeros((target_height, total_width, 3), dtype=np.uint8)
    
    # 将所有调整大小后的图像粘贴到新的图像上
    x_offset = 0
    for img in resized_images:
        merged_image[:, x_offset:x_offset + img.shape[1]] = img
        x_offset += img.shape[1]
    
    # 保存合并后的图像
    cv2.imwrite(output_path, merged_image)


def draw_show_img(img,result):
    polys = result['POLYGONS']
    centers = result['CENTER']
    angle_cls = result['LABELS']
    bbox = result['BBOX']
    for idx, poly in enumerate(polys):
        poly = poly.reshape(4, 2).astype(np.int32)
        ori_center = ((bbox[idx][0]+bbox[idx][2])//2,(bbox[idx][1]+bbox[idx][3])//2)
        color = (np.random.randint(0,255),np.random.randint(0,255),np.random.randint(0,255))
        img = cv2.drawContours(img,[poly],-1,color,2)
        img = cv2.circle(img,tuple(centers[idx].astype(np.int64).tolist()),5,color,thickness=2)
        img = cv2.circle(img,ori_center,5,(0,0,255),thickness=2)
        img = cv2.putText(img,str(angle_cls[idx]),ori_center,cv2.FONT_HERSHEY_SIMPLEX,2,(0,0,255),2)
    cv2.imwrite('./show.jpg',img)

def get_train_file(det_dir,pre_img_dir='./pre_img',pre_gt_dir='./pre_gt',model_path='./openvino/model.xml'):
    """
    det_dir = r'../yyzz2022/yyzz_detect_train/img_rotate'
    pre_gt_dir = r'./pre_gt'
    pre_img_dir = r'./pre_img'
    model_path = r'./openvino/model.xml'
    get_train_file(det_dir, pre_img_dir,pre_gt_dir, model_path)
    """
    import os
    from tqdm import tqdm
    from infer import CardRotate
    if not os.path.exists(pre_gt_dir):
        os.mkdir(pre_gt_dir)
    if not os.path.exists(pre_img_dir):
        os.mkdir(pre_img_dir)
    rotate_bin = CardRotate()
    rotate_bin.load_torch(model_path)
    files = os.listdir(det_dir)
    for file in tqdm(files):
        img = cv2.imread(os.path.join(det_dir,file))
        result = rotate_bin.rotate(img)
        with open(os.path.join(pre_gt_dir,os.path.splitext(file)[0]+'.txt'),'w+',encoding='utf-8') as fid:
            polys = result['POLYGONS'].astype(np.int32)
            angle_cls = result['LABELS']
            bbox = result['BBOX']
            is_photocopy = result['LAYOUT']
            tag = 0
            for (p,a,b,ip) in zip(polys,angle_cls,bbox,is_photocopy):
                p = ','.join([str(x) for x in p])
                b = ','.join([str(x) for x in b])
                fid.write(b+','+p+','+str(a)+','+str(ip)+'\n')
                tag += 1
        if tag==0:
            os.remove(os.path.join(pre_gt_dir,os.path.splitext(file)[0]+'.txt'))
        else:
            cv2.imwrite(os.path.join(pre_img_dir,os.path.splitext(file)[0]+'.jpg'),img)

def order_points_new(pts):
    # sort the points based on their x-coordinates
    xSorted = pts[np.argsort(pts[:, 0]), :]

    # grab the left-most and right-most points from the sorted
    # x-roodinate points
    leftMost = xSorted[:2, :]
    rightMost = xSorted[2:, :]
    if leftMost[0,1]!=leftMost[1,1]:
        leftMost=leftMost[np.argsort(leftMost[:,1]),:]
    else:
        leftMost=leftMost[np.argsort(leftMost[:,0])[::-1],:]
    (tl, bl) = leftMost
    if rightMost[0,1]!=rightMost[1,1]:
        rightMost=rightMost[np.argsort(rightMost[:,1]),:]
    else:
        rightMost=rightMost[np.argsort(rightMost[:,0])[::-1],:]
    (tr,br)=rightMost
    # return the coordinates in top-left, top-right,
    # bottom-right, and bottom-left order
    return np.array([tl, tr, br, bl], dtype="float32")

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        if self.count > 0:
            self.avg = self.sum / self.count

def get_package_installation_path(package_name):
    try:
        # 获取包的分布信息
        distribution = importlib.metadata.distribution(package_name)
        # 获取包的安装路径
        package_path = distribution.locate_file('')
        return package_path
    except importlib.metadata.PackageNotFoundError:
        print(f"包 '{package_name}' 未找到")
        return None
    except Exception as e:
        print(f"发生错误: {e}")
        return None


def transform_preds(coords, center, scale, output_size, rot=0):
    target_coords = np.zeros(coords.shape)
    trans = get_affine_transform(center, scale, rot, output_size, inv=1)
    for p in range(coords.shape[0]):
        target_coords[p, 0:2] = affine_transform(coords[p, 0:2], trans)
    return target_coords


def get_affine_transform(center,
                         scale,
                         rot,
                         output_size,
                         shift=np.array([0, 0], dtype=np.float32),
                         inv=0):
    if not isinstance(scale, np.ndarray) and not isinstance(scale, list):
        scale = np.array([scale, scale], dtype=np.float32)

    scale_tmp = scale
    src_w = scale_tmp[0]
    dst_w = output_size[0]
    dst_h = output_size[1]

    rot_rad = np.pi * rot / 180
    src_dir = get_dir([0, src_w * -0.5], rot_rad)
    dst_dir = np.array([0, dst_w * -0.5], np.float32)

    src = np.zeros((3, 2), dtype=np.float32)
    dst = np.zeros((3, 2), dtype=np.float32)
    src[0, :] = center + scale_tmp * shift
    src[1, :] = center + src_dir + scale_tmp * shift
    dst[0, :] = [dst_w * 0.5, dst_h * 0.5]
    dst[1, :] = np.array([dst_w * 0.5, dst_h * 0.5], np.float32) + dst_dir

    src[2:, :] = get_3rd_point(src[0, :], src[1, :])
    dst[2:, :] = get_3rd_point(dst[0, :], dst[1, :])

    if inv:
        trans = cv2.getAffineTransform(np.float32(dst), np.float32(src))
    else:
        trans = cv2.getAffineTransform(np.float32(src), np.float32(dst))

    return trans


def affine_transform(pt, t):
    new_pt = np.array([pt[0], pt[1], 1.0], dtype=np.float32).T
    new_pt = np.dot(t, new_pt)
    return new_pt[:2]


def get_dir(src_point, rot_rad):
    sn, cs = np.sin(rot_rad), np.cos(rot_rad)

    src_result = [0, 0]
    src_result[0] = src_point[0] * cs - src_point[1] * sn
    src_result[1] = src_point[0] * sn + src_point[1] * cs

    return src_result


def get_3rd_point(a, b):
    direct = a - b
    return b + np.array([-direct[1], direct[0]], dtype=np.float32)


def _sigmoid(x):
    y = torch.clamp(x.sigmoid_(), min=1e-4, max=1 - 1e-4)
    return y


def _gather_feat(feat, ind, mask=None):
    dim = feat.size(2)
    ind = ind.unsqueeze(2).expand(ind.size(0), ind.size(1), dim)
    feat = feat.gather(1, ind)
    if mask is not None:
        mask = mask.unsqueeze(2).expand_as(feat)
        feat = feat[mask]
        feat = feat.view(-1, dim)
    return feat


def _tranpose_and_gather_feat(feat, ind):
    feat = feat.permute(0, 2, 3, 1).contiguous()
    feat = feat.view(feat.size(0), -1, feat.size(3))
    feat = _gather_feat(feat, ind)
    return feat


def _nms(heat, kernel=3):
    pad = (kernel - 1) // 2
    hmax = nn.functional.max_pool2d(heat, (kernel, kernel), stride=1, padding=pad)
    keep = (hmax == heat).float()
    return heat * keep, keep


def _topk(scores, K=40):
    batch, cat, height, width = scores.size()

    topk_scores, topk_inds = torch.topk(scores.view(batch, cat, -1), K)

    topk_inds = topk_inds % (height * width)
    topk_ys = (topk_inds / width).int().float()
    topk_xs = (topk_inds % width).int().float()

    topk_score, topk_ind = torch.topk(topk_scores.view(batch, -1), K)
    topk_clses = (topk_ind / K).int()
    topk_inds = _gather_feat(topk_inds.view(batch, -1, 1),topk_ind).view(batch, K)
    topk_ys = _gather_feat(topk_ys.view(batch, -1, 1), topk_ind).view(batch, K)
    topk_xs = _gather_feat(topk_xs.view(batch, -1, 1), topk_ind).view(batch, K)

    return topk_score, topk_inds, topk_clses, topk_ys, topk_xs


def decode_by_ind(heat, inds, K=100):
    batch, cat, height, width = heat.size()
    score = _tranpose_and_gather_feat(heat, inds)
    score = score.view(batch, K, cat)
    _, Type = torch.max(score, 2)
    return Type


def bbox_decode(heat, wh, reg=None, K=100):
    batch, cat, height, width = heat.size()

    heat, keep = _nms(heat)

    scores, inds, clses, ys, xs = _topk(heat, K=K)
    if reg is not None:
        reg = _tranpose_and_gather_feat(reg, inds)
        reg = reg.view(batch, K, 2)
        xs = xs.view(batch, K, 1) + reg[:, :, 0:1]
        ys = ys.view(batch, K, 1) + reg[:, :, 1:2]
    else:
        xs = xs.view(batch, K, 1) + 0.5
        ys = ys.view(batch, K, 1) + 0.5
    wh = _tranpose_and_gather_feat(wh, inds)
    wh = wh.view(batch, K, 8)
    clses = clses.view(batch, K, 1).float()
    scores = scores.view(batch, K, 1)
    bboxes = torch.cat(
        [
            xs - wh[..., 0:1],
            ys - wh[..., 1:2],
            xs - wh[..., 2:3],
            ys - wh[..., 3:4],
            xs - wh[..., 4:5],
            ys - wh[..., 5:6],
            xs - wh[..., 6:7],
            ys - wh[..., 7:8],
        ],
        dim=2,
    )
    detections = torch.cat([bboxes, scores, clses, xs, ys], dim=2)

    return detections, inds


def gbox_decode(mk, st_reg, reg=None, K=400):
    batch, cat, height, width = mk.size()
    mk, keep = _nms(mk)
    scores, inds, clses, ys, xs = _topk(mk, K=K)
    if reg is not None:
        reg = _tranpose_and_gather_feat(reg, inds)
        reg = reg.view(batch, K, 2)
        xs = xs.view(batch, K, 1) + reg[:, :, 0:1]
        ys = ys.view(batch, K, 1) + reg[:, :, 1:2]
    else:
        xs = xs.view(batch, K, 1) + 0.5
        ys = ys.view(batch, K, 1) + 0.5
    scores = scores.view(batch, K, 1)
    clses = clses.view(batch, K, 1).float()
    st_Reg = _tranpose_and_gather_feat(st_reg, inds)
    bboxes = torch.cat(
        [
            xs - st_Reg[..., 0:1],
            ys - st_Reg[..., 1:2],
            xs - st_Reg[..., 2:3],
            ys - st_Reg[..., 3:4],
            xs - st_Reg[..., 4:5],
            ys - st_Reg[..., 5:6],
            xs - st_Reg[..., 6:7],
            ys - st_Reg[..., 7:8],
        ],
        dim=2,
    )
    return torch.cat([xs, ys, bboxes, scores, clses], dim=2), keep


def bbox_post_process(bbox, c, s, h, w):
    for i in range(bbox.shape[0]):
        bbox[i, :, 0:2] = transform_preds(bbox[i, :, 0:2], c[i], s[i], (w, h))
        bbox[i, :, 2:4] = transform_preds(bbox[i, :, 2:4], c[i], s[i], (w, h))
        bbox[i, :, 4:6] = transform_preds(bbox[i, :, 4:6], c[i], s[i], (w, h))
        bbox[i, :, 6:8] = transform_preds(bbox[i, :, 6:8], c[i], s[i], (w, h))
        bbox[i, :, 10:12] = transform_preds(bbox[i, :, 10:12], c[i], s[i], (w, h))
    return bbox


def gbox_post_process(gbox, c, s, h, w):
    for i in range(gbox.shape[0]):
        gbox[i, :, 0:2] = transform_preds(gbox[i, :, 0:2], c[i], s[i], (w, h))
        gbox[i, :, 2:4] = transform_preds(gbox[i, :, 2:4], c[i], s[i], (w, h))
        gbox[i, :, 4:6] = transform_preds(gbox[i, :, 4:6], c[i], s[i], (w, h))
        gbox[i, :, 6:8] = transform_preds(gbox[i, :, 6:8], c[i], s[i], (w, h))
        gbox[i, :, 8:10] = transform_preds(gbox[i, :, 8:10], c[i], s[i],
                                           (w, h))
    return gbox


def nms(dets, thresh):
    if len(dets) < 2:
        return dets
    index_keep = []
    keep = []
    for i in range(len(dets)):
        box = dets[i]
        if box[8] < thresh:
            break
        max_score_index = -1
        ctx = (dets[i][0] + dets[i][2] + dets[i][4] + dets[i][6]) / 4
        cty = (dets[i][1] + dets[i][3] + dets[i][5] + dets[i][7]) / 4
        for j in range(len(dets)):
            if i == j or dets[j][8] < thresh:
                break
            x1, y1 = dets[j][0], dets[j][1]
            x2, y2 = dets[j][2], dets[j][3]
            x3, y3 = dets[j][4], dets[j][5]
            x4, y4 = dets[j][6], dets[j][7]
            a = (x2 - x1) * (cty - y1) - (y2 - y1) * (ctx - x1)
            b = (x3 - x2) * (cty - y2) - (y3 - y2) * (ctx - x2)
            c = (x4 - x3) * (cty - y3) - (y4 - y3) * (ctx - x3)
            d = (x1 - x4) * (cty - y4) - (y1 - y4) * (ctx - x4)
            if (a > 0 and b > 0 and c > 0 and d > 0) or (a < 0 and b < 0
                                                         and c < 0 and d < 0):
                if dets[i][8] > dets[j][8] and max_score_index < 0:
                    max_score_index = i
                elif dets[i][8] < dets[j][8]:
                    max_score_index = -2
                    break
        if max_score_index > -1:
            index_keep.append(max_score_index)
        elif max_score_index == -1:
            index_keep.append(i)
    for i in range(0, len(index_keep)):
        keep.append(dets[index_keep[i]])
    return np.array(keep)


def group_bbox_by_gbox(bboxes,
                       gboxes,
                       score_thred=0.3,
                       v2c_dist_thred=2,
                       c2v_dist_thred=0.5):

    def point_in_box(box, point):
        x1, y1, x2, y2 = box[0], box[1], box[2], box[3]
        x3, y3, x4, y4 = box[4], box[5], box[6], box[7]
        ctx, cty = point[0], point[1]
        a = (x2 - x1) * (cty - y1) - (y2 - y1) * (ctx - x1)
        b = (x3 - x2) * (cty - y2) - (y3 - y2) * (ctx - x2)
        c = (x4 - x3) * (cty - y3) - (y4 - y3) * (ctx - x3)
        d = (x1 - x4) * (cty - y4) - (y1 - y4) * (ctx - x4)
        if (a > 0 and b > 0 and c > 0 and d > 0) or (a < 0 and b < 0 and c < 0
                                                     and d < 0):
            return True
        else:
            return False

    def get_distance(pt1, pt2):
        return math.sqrt((pt1[0] - pt2[0]) * (pt1[0] - pt2[0])
                         + (pt1[1] - pt2[1]) * (pt1[1] - pt2[1]))

    dets = copy.deepcopy(bboxes)
    sign = np.zeros((len(dets), 4))

    for idx, gbox in enumerate(gboxes):  # vertex x,y, gbox, score
        if gbox[10] < score_thred:
            break
        vertex = [gbox[0], gbox[1]]
        for i in range(0, 4):
            center = [gbox[2 * i + 2], gbox[2 * i + 3]]
            if get_distance(vertex, center) < v2c_dist_thred:
                continue
            for k, bbox in enumerate(dets):
                if bbox[8] < score_thred:
                    break
                if sum(sign[k]) == 4:
                    continue
                w = (abs(bbox[6] - bbox[0]) + abs(bbox[4] - bbox[2])) / 2
                h = (abs(bbox[3] - bbox[1]) + abs(bbox[5] - bbox[7])) / 2
                m = max(w, h)
                if point_in_box(bbox, center):
                    min_dist, min_id = 1e4, -1
                    for j in range(0, 4):
                        dist = get_distance(vertex,
                                            [bbox[2 * j], bbox[2 * j + 1]])
                        if dist < min_dist:
                            min_dist = dist
                            min_id = j
                    if (min_id > -1 and min_dist < c2v_dist_thred * m
                            and sign[k][min_id] == 0):
                        bboxes[k][2 * min_id] = vertex[0]
                        bboxes[k][2 * min_id + 1] = vertex[1]
                        sign[k][min_id] = 1
    return bboxes
