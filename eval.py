import os
import cv2
import numpy as np
from infer import CardRotate
from config import *
from tqdm import tqdm
from shapely.geometry import Polygon
from utils.utils import order_points_new


def compute_iou_polygon(poly1, poly2):
    """
    计算两个多边形（四边形）的交并比（IoU）。
    :param poly1: 四边形的顶点坐标，格式为 [(x1, y1), (x2, y2), (x3, y3), (x4, y4)]
    :param poly2: 四边形的顶点坐标
    :return: IoU值
    """
    # 创建多边形对象
    polygon1 = Polygon(poly1)
    polygon2 = Polygon(poly2)

    # 计算交集和并集
    intersection_area = polygon1.intersection(polygon2).area
    union_area = polygon1.union(polygon2).area

    # IoU 计算
    return intersection_area / union_area if union_area > 0 else 0


def cal_distance(poly1, poly2):
    poly1 = order_points_new(np.array(poly1))
    poly2 = order_points_new(np.array(poly2))
    diff = poly1 - poly2
    distance = np.sqrt(np.sum(diff ** 2, axis=1)).sum()
    return distance


def compute_metrics(predictions, ground_truths, iou_threshold=0.5):
    """
    计算模型性能指标，包括召回率、精确率和mAP。
    :param predictions: 模型预测结果, list of dicts
    :param ground_truths: 真实标签, list of dicts
    :param iou_threshold: IoU 阈值
    :return: 召回率, 精确率
    """
    tp = 0  # True Positives
    fp = 0  # False Positives
    fn = 0  # False Negatives
    
    coord_distance = 0
    for gt in ground_truths:
        detected = False
        for pred in predictions:
            iou = compute_iou_polygon(pred['poly'], gt['poly'])
            if iou >= iou_threshold and pred['class'] == gt['class']:
                tp += 1
                coord_distance += cal_distance(pred['poly'], gt['poly'])
                detected = True
                break
        if not detected:
            fn += 1
            
    fp = len(predictions) - tp

    # 计算召回率和精确率
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0

    return {
        'Recall': recall,
        'Precision': precision,
        'CornerDistance':coord_distance
    }

def eval(data_dir,gt_dir,iou_threshold):
    rotate_bin = CardRotate()
    if TEST_TYPE == "test_torch":
        rotate_bin.load_torch(TEST_MODEL_PATH)
    elif TEST_TYPE == "test_openvino":
        rotate_bin.load_openvino(TEST_MODEL_PATH)
    files = os.listdir(data_dir)
    
    bar = tqdm(total=len(files))
    
    total_metrics = {
            'Recall': 0,
            'Precision': 0,
            'CornerDistance':0
            }
    
    for i in range(len(files)): 
        file = files[i]
        bar.update(1)
        gt = []
        with open(os.path.join(gt_dir, file.split('.')[0] + '.txt'),"r",encoding='utf-8') as fid:
            for line in fid.readlines():
                line = line.strip()
                line = list(map(float,line.split(',')))
                gt.append(line)
        img = cv2.imread(os.path.join(data_dir,file))
        out = rotate_bin.rotate(img)
        gts = []
        pres = []
        for item in gt:
            gts.append({"poly":np.array(item[4:12]).reshape(4,2).tolist(),"class":item[12]})
        for poly,cls in zip(out['POLYGONS'],out['LABELS']):
            pres.append({"poly":poly.reshape(4,2).tolist(),"class":cls})

        metrics = compute_metrics(pres,gts,iou_threshold=iou_threshold)
        total_metrics['Recall'] += metrics['Recall']
        total_metrics['Precision'] += metrics['Precision']
        total_metrics['CornerDistance'] += metrics['CornerDistance']
        
    bar.close()
    total_metrics['Recall'] = total_metrics['Recall']/len(files)
    total_metrics['Precision'] = total_metrics['Precision']/len(files)
    total_metrics['CornerDistance'] = total_metrics['CornerDistance']/len(files) 
    
    return total_metrics
        
metrics = eval(EVAL_DIR,EVAL_GT,IOU_THRESH)
print("Recall:{}\tPrecision:{}\tCornerDistance:{}".format(metrics['Recall'],metrics['Precision'],metrics['CornerDistance']))
with open("eval_{}_metrics.txt".format(TEST_MODEL_PATH.split('/')[-1]),'w+') as fid:
    fid.write(str(metrics))
        
        
        