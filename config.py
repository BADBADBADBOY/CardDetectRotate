#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Author : BADBADBADBOY
# @File : config.py
# @Time : 2024/8/20 21:22

### data
# gt_dir = './pre_gt' ### 标签文件地址
# img_dir = './pre_img' ### 训练图片地址

gt_dir = '/src/notebooks/SyntheticCards_train100k/gt_dir' ### 标签文件地址
img_dir = '/src/notebooks/SyntheticCards_train100k/data' ### 训练图片地址

MAX_OBJS = 8 ## 这个最好调成统计下所有图里面的目标数，取最大值，因为训练时候我的一张图只有2-3个目标，所以调成了8
INPUT_H = 768 ### 模型训练时，训练图片的高
INPUT_W = 768 ### 模型训练时，训练图片的宽

### train
MODEL_SUPPORT = ['resnet','lcnet','replcnet']
NEED_FTYPE = False

# BASE_MODEL = "resnet"
# NUM_LAYER = 18

BASE_MODEL = "lcnet"
MODEL_RATIO = 0.3

# BASE_MODEL = "replcnet"
# MODEL_RATIO = 1.

LEARNING_RATE= 0.001 ### 初始学习率
NUM_EPOCH = 200  ### 训练的epoch数
BATCH_SIZE = 64  ### 训练的batch_size
NUM_WORKER = 4 ### num_worker
SHOW_STEP = 100 ### 迭代多少次打印一次log
SAVE_ITERS = 1000 ### 迭代多少次保存一次网络参数
SAVE_PATH = "./checkpoint_" + f"{BASE_MODEL}" ### 网络参数存储的地址

HM_WEIGHT = 5 ### 中心点loss的权重
WH_WEIGHT = 1 ### 四个角点回归loss的权重
OFFSET_WEIGHT = 1 ### 中心点偏移loss的权重
ANGLE_WEIGHT = 1 ### 4角度分类loss的权重
TYPE_WEIGHT = 1 ### 类别分类的权重

LOAD_PRE_MODEL = False ### 是否加载预训练网络参数

# LOAD_TYPE = 'resnet18' ### 加载resnet18作为预训练参数
# PRE_MODEL_PATH = "/src/notebooks/CardRotation/resnet18-5c106cde.pth"

# LOAD_TYPE = 'modelscope' ### 加载此链接模型作为权重初始参数：https://modelscope.cn/models/iic/cv_resnet18_card_correction
# PRE_MODEL_PATH = "./pytorch_model.pt"

# LOAD_TYPE = 'lcnet' ### 加载resnet18作为预训练参数
# PRE_MODEL_PATH = "./checkpoint_replcnet/model_12000.pth"

OPTIM_METHOD = 'Adam' # SGD ### 优化器设置，建议用Adam

### 优化器设置
OPTIM_CONFIG = {}
OPTIM_CONFIG['optimizer'] = {}
OPTIM_CONFIG['optimizer']['base_lr'] = LEARNING_RATE
OPTIM_CONFIG['optimizer']['n_epoch'] = NUM_EPOCH
OPTIM_CONFIG['optimizer']['n_epoch'] = NUM_EPOCH
OPTIM_CONFIG['optimizer']['factor'] = 0.9
OPTIM_CONFIG['optimizer']['weight_decay'] = 5e-5
OPTIM_CONFIG['optimizer']['momentum'] = 0.9


### test

EVAL_DIR = "/src/notebooks/SyntheticCards_val1k/data" ### 验证集的图片地址
EVAL_GT = "/src/notebooks/SyntheticCards_val1k/gt_dir" ### 验证集的标注地址
IOU_THRESH = 0.4 ### IOU阈值

K = 10 ### 保留前K个预测中心点，这里如果你检测的目标特别多，需要加大，因为设置10只一张图只能检测10个目标
TEST_H = 768 ###模型测试时，测试图片的高，这里得是64的整数倍
TEST_W = 768 ###模型测试时，测试图片的宽，这里得是64的整数倍
OBJ_SCORE = 0.5 ### 过滤中心点的阈值，低于这分数的中心点不要
USE_GPU = False ### 测试时是否用GPU

# ### 使用此链接模型作为权重测试：https://modelscope.cn/models/iic/cv_resnet18_card_correction
# TEST_TYPE = "test_torch" ### 共三种模式，test_torch, test_openvino, torch2openvino
# TEST_LOAD_TYPE = 'modelscope'
# TEST_MODEL_PATH = "./pytorch_model.pt"

### 使用自己训练的模型做测试
TEST_TYPE = "test_torch" ### 共三种模式，test_torch, test_openvino, torch2openvino
TEST_LOAD_TYPE = 'trained'
TEST_MODEL_PATH = './checkpoint_{}/model_48000.pth'.format(BASE_MODEL)

# ### 将此链接模型作为权重的模型转成openvino: https://modelscope.cn/models/iic/cv_resnet18_card_correction
# TEST_LOAD_TYPE = 'modelscope'
# TEST_MODEL_PATH = "./pytorch_model.pt"
# TEST_TYPE = "torch2openvino" ### 共三种模式，test_torch, test_openvino, torch2openvino
# USE_FP16 = True ### 转换时是否保存成float16

# ### 将自己训练的模型转成openvino
# TEST_LOAD_TYPE = 'trained'
# TEST_MODEL_PATH = './checkpoint/model_200.pth'
# TEST_TYPE = "torch2openvino" ### 共三种模式，test_torch, test_openvino, torch2openvino
# USE_FP16 = True ### 转换时是否保存成float16

# ### 使用openvino做预测可加速
# TEST_TYPE = "test_openvino" ### 共三种模式，test_torch, test_openvino, torch2openvino
# TEST_MODEL_PATH = './openvino/model.xml'





