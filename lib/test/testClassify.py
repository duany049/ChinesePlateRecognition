# coding=utf-8

import matplotlib.pyplot as plt
import numpy as np
from lib.common.config import cfg
import argparse
import os

NETS = {'vgg16': ('vgg16_faster_rcnn_iter_70000.ckpt',), 'res101': ('res101_faster_rcnn_iter_110000.ckpt',)}
DATASETS = {'pascal_voc': ('voc_2007_trainval',), 'pascal_voc_0712': ('voc_2007_trainval+voc_2012_trainval',)}


def parse_args():
    """解析输入参数"""
    parser = argparse.ArgumentParser(description='test classify code')
    parser.add_argument('--net', dest='model_net', help='Model to use [vgg16 res101]',
                        choices=NETS.keys(), default='vgg16')
    # todo 改成车牌
    parser.add_argument('--dataset', dest='dataset', help='Trained dataset [pascal_voc pascal_voc_0712]',
                        choices=DATASETS.keys(), default='pascal_voc_0712')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    # cfg.TEST.HAS_RPN = true

    # path
    model_net = args.model_net
    dataset = args.dataset
    model = os.path.join('output', model_net, DATASETS[dataset][0], 'default',
                         NETS[model_net][0])
    if not os.path.isfile(model + '.meta'):
        raise IOError(('{:s} not found.\nDid you download the proper networks from '
                       'our server and place them properly?').format(model + '.meta'))

