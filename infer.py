#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Author : BADBADBADBOY
# @File : infer.py
# @Time : 2024/8/20 21:22

import time
import os
import cv2
import torch
import shutil
import math
import numpy as np
from utils.utils import get_affine_transform,bbox_decode,decode_by_ind,nms,bbox_post_process,get_package_installation_path
from openvino.runtime import Core, AsyncInferQueue
from utils.utils import order_points_new,draw_show_img,merge_images_horizontal
from config import *

class CardRotate(object):
    K = K
    input_h, input_w = TEST_H, TEST_W

    def load_torch(self,model_path):
        super(CardRotate,self).__init__()
        if BASE_MODEL == "resnet":
            from model.models_resnet import CardDetectionCorrectionModel
            model = CardDetectionCorrectionModel(num_layers=NUM_LAYER)
        elif BASE_MODEL == "lcnet":
            from model.models_lcnet import CardDetectionCorrectionModel
            model = CardDetectionCorrectionModel(ratio=MODEL_RATIO)

        self.infer_model = model
        if TEST_LOAD_TYPE == 'modelscope' and BASE_MODEL=='resnet':
            model_dict = torch.load(model_path, map_location='cpu')['state_dict']
        else:
            model_dict = torch.load(model_path, map_location='cpu')
        self.infer_model.load_state_dict(model_dict)
        self.infer_model.eval()
        self.infer_type = 'torch'
        if torch.cuda.is_available() and USE_GPU:
            self.infer_model = self.infer_model.cuda()

    def load_openvino(self,openvino_model_path):
        ie = Core()
        model_ir = ie.read_model(model=openvino_model_path)
        try:
            compiled_model_ir = ie.compile_model(model=model_ir, device_name="CPU",
                                                 config={"PERFORMANCE_HINT": "LATENCY",
                                                         "CPU_RUNTIME_CACHE_CAPACITY": "0",
                                                         'CPU_THREADS_NUM': '4'})
        except:
            compiled_model_ir = ie.compile_model(model=model_ir, device_name="CPU",
                                                 config={"PERFORMANCE_HINT": "LATENCY",
                                                         "CPU_RUNTIME_CACHE_CAPACITY": "0"
                                                         })
        self.infer_request = compiled_model_ir.create_infer_request()
        self.infer_type = 'openvino'

    def rotate(self,img):
        pre_input = self.preprocess(img)
        if self.infer_type == 'torch':   
            pre_out = self.forward(pre_input)
        elif self.infer_type == 'openvino':
            pre_out = self.forward_openvino(pre_input)
            for key in pre_out['results'][0].keys():
                pre_out['results'][0][key] = torch.from_numpy(pre_out['results'][0][key])
        out = self.postprocess(pre_out)
        return out

    def onnx_export(self,img,onnx_path):
        torch.onnx.export(self.infer_model,  # 导出的模型
                          self.preprocess(img)['img'],  # 输入数据
                          onnx_path,  # 保存的文件名
                          export_params=True,  # 是否导出模型参数
                          opset_version=11,  # ONNX 版本号
                          do_constant_folding=True,  # 是否执行常量折叠优化
                          input_names=['input'],  # 输入名
                          output_names=['output'])  # 输出名

    def onnx2openvino(self,use_fp16 = True):
        os.system(r"python {} "
                  "--input_model {}  "
                  "--input_shape [-1,3,-1,-1]  "
                  "--compress_to_fp16 {} "
                  "--output_dir {}".format(os.path.join(get_package_installation_path("openvino"),'openvino/tools/mo/mo.py'),'./model.onnx',use_fp16,'./openvino'))

    def preprocess(self, img):
        self.image = img
        mean = np.array([0.408, 0.447, 0.470],dtype=np.float32).reshape(1, 1, 3)
        std = np.array([0.289, 0.274, 0.278],dtype=np.float32).reshape(1, 1, 3)
        height, width = img.shape[0:2]
        inp_height, inp_width = self.input_h, self.input_w
        c = np.array([width / 2., height / 2.], dtype=np.float32)
        s = max(height, width) * 1.0
        trans_input = get_affine_transform(c, s, 0, [inp_width, inp_height])
        resized_image = cv2.resize(img, (width, height))
        inp_image = cv2.warpAffine(
            resized_image,
            trans_input, (inp_width, inp_height),
            flags=cv2.INTER_LINEAR)
        inp_image = ((inp_image / 255. - mean) / std).astype(np.float32)
        images = inp_image.transpose(2, 0, 1).reshape(1, 3, inp_height,inp_width)
        if self.infer_type == 'torch':
            images = torch.from_numpy(images)
            if torch.cuda.is_available() and USE_GPU:
                images = images.cuda()
        meta = {
            'c': c,
            's': s,
            'input_height': inp_height,
            'input_width': inp_width,
            'out_height': inp_height // 4,
            'out_width': inp_width // 4
        }

        result = {'img': images, 'meta': meta}
        return result

    def distance(self, x1, y1, x2, y2):
        return math.sqrt(pow(x1 - x2, 2) + pow(y1 - y2, 2))

    def crop_image(self, img, position):
        x0, y0 = position[0][0], position[0][1]
        x1, y1 = position[1][0], position[1][1]
        x2, y2 = position[2][0], position[2][1]
        x3, y3 = position[3][0], position[3][1]

        img_width = self.distance((x0 + x3) / 2, (y0 + y3) / 2, (x1 + x2) / 2,
                                  (y1 + y2) / 2)
        img_height = self.distance((x0 + x1) / 2, (y0 + y1) / 2, (x2 + x3) / 2,
                                   (y2 + y3) / 2)

        corners_trans = np.zeros((4, 2), np.float32)
        corners_trans[0] = [0, 0]
        corners_trans[1] = [img_width, 0]
        corners_trans[2] = [img_width, img_height]
        corners_trans[3] = [0, img_height]

        transform = cv2.getPerspectiveTransform(position, corners_trans)
        dst = cv2.warpPerspective(img, transform,
                                  (int(img_width), int(img_height)))
        return dst

    def forward(self, input) :
        pred = self.infer_model(input['img'])
        return {'results': pred, 'meta': input['meta']}

    def forward_openvino(self,input):
        self.infer_request.infer([input['img']])
        openvino_output = {}
        openvino_output['hm'] = self.infer_request.get_output_tensor(0).data
        openvino_output['cls'] = self.infer_request.get_output_tensor(1).data
        openvino_output['ftype'] = self.infer_request.get_output_tensor(2).data
        openvino_output['wh'] = self.infer_request.get_output_tensor(3).data
        openvino_output['reg'] = self.infer_request.get_output_tensor(4).data
        return {'results': [openvino_output], 'meta': input['meta']}

    def postprocess(self, inputs):
        output = inputs['results'][0]
        meta = inputs['meta']
        wh = output['wh']
        reg = output['reg']
        
        hm = output['hm'].sigmoid_()
        angle_cls = output['cls'].sigmoid_()
        ftype_cls = output['ftype'].sigmoid_()

        bbox, inds = bbox_decode(hm, wh, reg=reg, K=self.K)
        angle_cls = decode_by_ind(angle_cls, inds, K=self.K).detach().cpu().numpy()
        ftype_cls = decode_by_ind(ftype_cls, inds,K=self.K).detach().cpu().numpy().astype(np.float32)
        bbox = bbox.detach().cpu().numpy()

        for i in range(bbox.shape[1]):
            bbox[0][i][9] = angle_cls[0][i]
        bbox = np.concatenate((bbox, np.expand_dims(ftype_cls, axis=-1)),axis=-1)
        bbox = nms(bbox, 0.3)
        bbox = bbox_post_process(bbox.copy(), [meta['c']],[meta['s']], meta['out_height'],meta['out_width'])
        res = []
        angle = []
        sub_imgs = []
        ftype = []
        score = []
        center = []
        corner_left_right = []
        for idx, box in enumerate(bbox[0]):
            if box[8] > OBJ_SCORE:
                angle.append(int(box[9]))
                res.append(box[0:8])
                box8point = np.array(box[0:8]).reshape(4,2).astype(np.int32)
                corner_left_right.append([box8point[:,0].min(),box8point[:,1].min(),box8point[:,0].max(),box8point[:,1].max()])
                sub_img = self.crop_image(self.image,res[-1].copy().reshape(4, 2))
                if angle[-1] == 1:
                    sub_img = cv2.rotate(sub_img, 2)
                if angle[-1] == 2:
                    sub_img = cv2.rotate(sub_img, 1)
                if angle[-1] == 3:
                    sub_img = cv2.rotate(sub_img, 0)
                sub_imgs.append(sub_img)
                ftype.append(int(box[12]))
                score.append(box[8])
                center.append([box[10],box[11]])

        result = {
            "POLYGONS": np.array(res),
            "BBOX": np.array(corner_left_right),
            "SCORES": np.array(score),
            "OUTPUT_IMGS": sub_imgs,
            "LABELS": np.array(angle),
            "LAYOUT": np.array(ftype),
            "CENTER": np.array(center)
        }
        return result



if __name__ == "__main__":
    
    img = cv2.imread(r'./pp4.jpg')
    
    rotate_bin = CardRotate()
    
    if TEST_TYPE == "test_torch":
        rotate_bin.load_torch(TEST_MODEL_PATH)
    elif TEST_TYPE == "test_openvino":
        rotate_bin.load_openvino(TEST_MODEL_PATH)
    elif TEST_TYPE == "torch2openvino":
        rotate_bin.load_torch(TEST_MODEL_PATH)
        print("torch模型转换onnx模型中......")
        rotate_bin.onnx_export(img,"./model.onnx")
        print("onnx模型转换完成 ！！！")
        print("onnx模型转换openvino模型中......")
        rotate_bin.onnx2openvino(use_fp16=USE_FP16)
        print("openvino模型转换成功 ！！！")
        os.remove("./model.onnx")
    if TEST_TYPE == "test_torch" or TEST_TYPE == "test_openvino":
        t_sum = 0
        for i in range(10):
            t = time.time()
            out = rotate_bin.rotate(img)
            t_sum += (time.time() - t)
        print("time avg:{}".format(t_sum/10))
        draw_show_img(img.copy(), out)
        merge_images_horizontal([img] + out['OUTPUT_IMGS'],"./pp4_rotate_show.jpg")
        cv2.imwrite(r'./rotate_img.jpg',out['OUTPUT_IMGS'][0])
