# -*- coding: utf-8 -*-
# @Time    : 2024/4/18 9:07
# @Author  : yblir
# @File    : clip_trt_infer.py
# explain  : 
# ======================================================================================================================
import logging
import os
import argparse

import cv2
import torch
from PIL import Image
from pathlib2 import Path
import numpy as np
import tensorrt as trt
from tensorrt import Logger, Runtime
import cn_clip.clip as clip
from cn_clip.clip.utils import _MODEL_INFO
from tensorrt_utils import TensorRTShape, build_engine, TensorRTModel
from torchvision.transforms import Compose, ToTensor, Normalize, Resize, InterpolationMode


class ChineseClip(object):
    def __init__(self, batch_size, context_length, text_onnx_path, img_onnx_path, fp16=True, input_size=224):
        self.batch_size = batch_size
        self.fp16 = fp16
        self.fp_flag = "fp16" if self.fp16 else "fp32"

        self.seq_len = context_length
        self.text_onnx_path = text_onnx_path
        self.img_onnx_path = img_onnx_path
        self.input_size = input_size

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "")

        # 在当前目录生成，保存trt engine
        self.text_trt_path = self.text_onnx_path.replace(".onnx", f"_{self.fp_flag}.trt")
        self.img_trt_path = self.img_onnx_path.replace(".onnx", f"_{self.fp_flag}.trt")
        self.create_engine()

        # 加载图像trt engine
        self.img_trt_engine = TensorRTModel(self.img_trt_path)
        # 加载文本trt engine
        self.text_trt_engine = TensorRTModel(self.text_trt_path)

    def create_engine(self):
        # 若engine已经存在，那就没必要再继续了
        if os.path.exists(self.text_trt_path) and os.path.exists(self.img_trt_path):
            logging.info("both of text and img trt engine are exists")
            return
        trt_logger: Logger = trt.Logger(trt.Logger.INFO)
        runtime: Runtime = trt.Runtime(trt_logger)
        trt.init_libnvinfer_plugins(trt_logger, '')

        # 配置文本shape
        text_input_shape = [TensorRTShape((self.batch_size, self.seq_len),
                                          (self.batch_size, self.seq_len),
                                          (self.batch_size, self.seq_len), 'text')]
        assert os.path.exists(
                self.text_onnx_path), f"Error: The specified --text-onnx-path {self.text_onnx_path} not exists!"

        # 配置图片shape
        vision_input_shape = [TensorRTShape((self.batch_size, 3, self.input_size, self.input_size),
                                            (self.batch_size, 3, self.input_size, self.input_size),
                                            (self.batch_size, 3, self.input_size, self.input_size), 'image')]
        assert os.path.exists(
                self.img_onnx_path), f"Error: The specified --vision-onnx-path {self.img_onnx_path} not exists!"

        logging.info("---------------build TensorRT engine: start------------------")
        # 构建引擎
        text_engine = build_engine(
                runtime=runtime,
                onnx_file_path=self.text_onnx_path,
                logger=trt_logger,
                input_shapes=text_input_shape,
                workspace_size=10000 * 1024 * 1024,
                fp16=self.fp16,
                int8=False,
        )
        with open(self.text_trt_path, 'wb') as f:
            f.write(bytearray(text_engine.serialize()))
        print(f"Saved the text {self.fp_flag.upper()} TensorRT model at {self.text_trt_path} ...")

        img_engine = build_engine(
                runtime=runtime,
                onnx_file_path=self.img_onnx_path,
                logger=trt_logger,
                input_shapes=vision_input_shape,
                workspace_size=10000 * 1024 * 1024,
                fp16=self.fp16,
                int8=False,
        )
        with open(self.img_trt_path, 'wb') as f:
            f.write(bytearray(img_engine.serialize()))
        print(f"Saved the vision {self.fp_flag.upper()} TensorRT model at {self.img_trt_path} ...")

    def img_trt_infer(self, img):
        # 用TensorRT模型计算图像侧特征
        image_features = self.img_trt_engine(inputs={'image': img})['unnorm_image_features']  # 未归一化的图像特征
        image_features /= image_features.norm(dim=-1, keepdim=True)  # 归一化后的Chinese-CLIP图像特征，用于下游任务
        # print(image_features.shape)  # Torch Tensor shape: [1, 特征向量维度]
        return image_features

    def text_trt_infer(self, text_list):
        texts = clip.tokenize(text_list, context_length=self.seq_len).cuda(self.device)
        text_features = self.text_trt_engine(inputs={'text': texts})['unnorm_text_features']
        # # 归一化后的Chinese-CLIP文本特征，用于下游任务
        text_features = text_features / text_features.norm(dim=1, keepdim=True)
        # print(text_features.shape)  # Torch Tensor shape: [4, 特征向量维度]
        return text_features


def image_transform(image_size=224):
    transform = Compose([
        Resize((image_size, image_size), interpolation=InterpolationMode.BICUBIC),
        ToTensor(),
        Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])
    return transform


def image_text_similarity(img_features, monitor_features, length_ca,
                          filter_features, distance_fi,
                          text_features, distance_te, distance_ca,
                          distance_cam, distance_re, distance_fu, distance_result
                          ):
    """
    计算图片与文本相似度,实现跨模态检索
    :param img_features: 从图片端输出的特征,[n,512],n:图片数量
    :param monitor_features: 布控数据特征,如卡通(n1,512),图片特征(n2,512)等,cat在一起后再转置,用于后续乘法,b=n1+n2,shape=(512,b)
    :param text_features: (b1,512),布控的文本特征
    :param filter_features: 过滤图片特征,即当前检索图片与这里面特征相似度高时, 当前图片跳过,(512,b2)
    :param distance_fi: 过滤相似度阈值
    :param length_ca:
    :param distance_te: 
    :param distance_ca: 
    :param distance_cam: 
    :param distance_re: 
    :param distance_fu: 
    :param distance_result: 
    :return: 
    """
    # 暂时假定monitor_features由卡通+图片特则组成,数量分别为n1,n2
    n1, n2 = 100, 50
    # (n,b), 计算当前图片与所有布控图片相似度
    img_similarities = torch.matmul(img_features, monitor_features)

    # 找出卡通,图片中相似度最大和平均值
    cartoon_max_similarity = torch.max(img_similarities[:, :n1], dim=1)[0].numpy()  # 每张图片与所有卡通,最大相似度值
    real_max_similarity = torch.max(img_similarities[:, n1:], dim=1)[0].numpy()  # 每张图片与所有照片,最大相似度值
    cartoon_mean_similarity = torch.mean(img_similarities[:, :n1], dim=1).numpy()  # 平均相似度
    real_mean_similarity = torch.mean(img_similarities[:, n1:], dim=1).numpy()      # 平均相似度

    # (n,b2), 计算当前图片与所有过滤图片相似度
    filter_similarities = torch.matmul(img_features, filter_features)
    # # 每张图片与所有过滤图片最大相似度
    filter_max_similarity = torch.max(filter_similarities, dim=1)[0].numpy()    
    # 过滤掩码,当图片与所有过滤图片中相似度最大的值都小于过滤阈值时,认为当前图片不能被过滤掉,保留进行下一步,[true,false,true,...]
    filter_img_mask = filter_max_similarity <= distance_fi
    
    # (n,b1),计算当前图片特征与文本特征相似度
    img_text_similarities = torch.matmul(img_features, text_features)
    # (n,), 找出每张图片与所有文本最大相似度
    img_text_max_similarity = torch.max(img_text_similarities, dim=1)[0].numpy()
    # 当图文最大相似度大于阈值,认为当前图片不能被忽略,保留进行下一步
    text_mask = img_text_max_similarity > distance_te
    # 保留经过 过滤图片,文本 都满足阈值的图片
    img_text_mask = filter_img_mask * text_mask
    # 再求平均相似度,如果布控图片只有一种, 不需要这步
    row_max_all = (cartoon_mean_similarity + real_mean_similarity) / 2.0

    similar_values = np.zeros_like(row_max_all, dtype=float)
    # 若有张图片与布控图片相似度最大值(不关心是布控图片的哪张) 大于阈值,保留
    cartoon_max_mask = cartoon_max_similarity >= distance_ca
    # 若有张图片与布控图片相似度平均值(不关心是布控图片的哪张) 大于阈值,保留
    cartoon_mean_mask = cartoon_mean_similarity >= distance_cam
    # 同上
    real_max_mask = real_max_similarity >= distance_fu
    real_mean_mask = real_mean_similarity >= distance_re
    
    # 二者中间段取值,why?
    control_distance_me = cartoon_max_mask * real_mean_mask
    control_distance_me = control_distance_me * img_text_mask
    control_mas_max = real_max_mask * img_text_mask
    cartoon_mean_mask = cartoon_mean_mask * img_text_mask
    if len(row_max_all[control_distance_me]) or len(real_max_similarity[control_mas_max]) or \
            len(cartoon_max_similarity[cartoon_mean_mask]):
        similar_values = np.where(control_mas_max, real_max_similarity, 0)
        indices_to_replace = np.where(np.logical_and(similar_values == 0, cartoon_mean_mask))
        similar_values[indices_to_replace] = np.array(cartoon_max_similarity)[indices_to_replace]
        indices_to_replace = np.where(np.logical_and(similar_values == 0, control_distance_me))
        similar_values[indices_to_replace] = np.array(row_max_all)[indices_to_replace]
        similar_values = np.where(similar_values < distance_result, 0, similar_values)
    return similar_values.tolist()


if __name__ == '__main__':
    ch_clip = ChineseClip(batch_size=1, context_length=52,
                          text_onnx_path="../run_codes.txt.fp16.onnx",
                          img_onnx_path="../run_codes.img.fp16.onnx",
                          fp16=True, input_size=224)
    img_transform = image_transform(image_size=224)

    img = Image.open("../examples/pokemon.jpeg")
    img = img_transform(img).unsqueeze(0).cuda()

    res = ch_clip.img_trt_infer(img)
    print(res.shape)
    print(res)
