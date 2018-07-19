# coding=utf-8

import cv2
from PIL import Image, ImageDraw
import os
import matplotlib.pyplot as plt
import numpy as np
import argparse
import random
import time
import shutil


def _generate_format_imgs(format_src_imgs, images_src_dir, images_out_dir):
    labels = np.empty((0,))
    coors_list = np.empty((0, 0), dtype=np.int32)
    index = 0
    for image_name in format_src_imgs:
        label, coors = _generate_format_img(image_name)
        if label is None:
            print('ignore None result in generate format img')
            continue
        dest_name = ("%06d" % index) + '.jpg'
        index += 1
        img_src_all_path = os.path.join(images_src_dir, image_name)
        img_out_all_path = os.path.join(images_out_dir, dest_name)
        print('img out all path: ', img_out_all_path)
        Image.open(img_src_all_path).save(img_out_all_path)
        labels = np.append(labels, label)

        if coors_list.shape[0] == 0:
            coors_list = np.append(coors_list, coors)
            coors_list = coors_list[np.newaxis, :]
        else:
            coors_list = np.vstack((coors_list, coors))
    return labels, coors_list


def _generate_format_img(image_name):
    """
    左上坐标：x,y分别取四个坐标中x,y最小值
    右下坐标：x,y分别取四个坐标中x,y最大值
    :return: 车牌内容以及左上右下的坐标
    """
    if not '_' in image_name:
        print('ignore the file, because file name: {%s} is invalid' % image_name)
        return None, None
    label_splits = image_name.split('_')
    if len(label_splits) > 2:
        print('ignore the file,because the file name {} has more than one \'_\'' % image_name)
        return None, None
    label = label_splits[0]
    other = label_splits[1]
    coors = other.split('-')
    if len(coors) != 8:
        print('ignore next coor action, because the coor: {0} is invalid'.format(coors))
        return None, None
    x1 = int(coors[0])
    y1 = int(coors[1])
    x2 = int(coors[2])
    y2 = int(coors[3])
    x3 = int(coors[4])
    y3 = int(coors[5])
    x4 = int(coors[6])
    y4 = coors[7].split('.')
    if len(y4) != 2:
        print('ignore next y4 action, because y4: {0} is invalid'.format(y4))
        return None, None
    y4 = int(y4[0])
    x_start = min([x1, x2, x3, x4])
    y_start = min([y1, y2, y3, y4])
    x_end = max([x1, x2, x3, x4])
    y_end = max([y1, y2, y3, y4])
    if x_start >= x_end or y_start >= y_end:
        print(
            'ignore next action,because coor value is invalid - x_start: {0} - y_start: {1} - x_end: {2} - y_end: {3}'
                .format(x_start, y_start, x_end, y_end))
        return None, None
    print(
        'label: {0} - x_start: {1} - y_start: {2} - x_end: {3} - y_end: {4}'.format(label, x_start, y_start, x_end,
                                                                                    y_end))
    return label, np.asarray((x_start, y_start, x_end, y_end))


def _write_img_info(labels, coors_list, out_dir):
    images_infos = os.path.join(out_dir, 'infos.txt')
    labels = labels[:, np.newaxis]
    imgs_info = np.hstack((labels, coors_list))
    with open(os.path.join(images_infos), 'wb') as f:
        for img_info in imgs_info:
            _write_info(f, img_info[0])
            _write_info(f, ' ')
            _write_info(f, img_info[1])
            _write_info(f, ' ')
            _write_info(f, img_info[2])
            _write_info(f, ' ')
            _write_info(f, img_info[3])
            _write_info(f, ' ')
            _write_info(f, img_info[4])
            f.write(b'\n')


def _write_info(f, info):
    f.write(info.encode('utf-8'))


def _parse_args():
    parser = argparse.ArgumentParser(description='Transform the format of imgs')
    parser.add_argument('--src_dir', dest='src_dir', default='imgs_src')
    parser.add_argument('--out_dir', dest='out_dir', default='transformed_imgs')
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = _parse_args()
    cur_path = os.path.dirname(__file__)
    imgs_src_dir = os.path.join(cur_path, args.src_dir)
    images_out_dir = os.path.join(cur_path, args.out_dir, )
    images_train_out_dir = os.path.join(images_out_dir, 'train')
    images_test_out_dir = os.path.join(images_out_dir, 'test')
    imgs_name_list = os.listdir(imgs_src_dir)
    if os.path.isdir(images_train_out_dir):
        shutil.rmtree(images_train_out_dir)
    os.makedirs(images_train_out_dir)
    if os.path.isdir(images_test_out_dir):
        shutil.rmtree(images_test_out_dir)
    os.makedirs(images_test_out_dir)
    start_time = time.time()
    random.shuffle(imgs_name_list)
    print('shuffle imgs use time: ', time.time() - start_time)
    train_ratio = 0.8
    train_num = int(len(imgs_name_list) * train_ratio)
    train_imgs = imgs_name_list[:train_num]
    test_imgs = imgs_name_list[train_num:]

    train_labels, train_coors_list = _generate_format_imgs(train_imgs, imgs_src_dir, images_train_out_dir)
    _write_img_info(train_labels, train_coors_list, images_train_out_dir)
    test_labels, test_coors_list = _generate_format_imgs(test_imgs, imgs_src_dir, images_test_out_dir)
    _write_img_info(test_labels, test_coors_list, images_test_out_dir)
