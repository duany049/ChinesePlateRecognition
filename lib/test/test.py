# coding=utf-8
import numpy as np
import tensorflow as tf
import _init_paths_test

# 验证imdb读取的正确性
from lib.datasets.standard_car_plate import standard_car_plate
standard_car_plate = standard_car_plate('train')
roidb = standard_car_plate.gt_roidb()
print(roidb)

# 测试分类代码是否正确


# 测试车牌分类代码是否正确


# 测试标签生成代码是否正确


# 测试车牌检测代码是否正确


# 测试端到端车牌检测和识别代码
