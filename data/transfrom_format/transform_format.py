# coding=utf-8

import cv2
from PIL import Image, ImageDraw
import os
import matplotlib.pyplot as plt


# def test():
#     """
#     测试原图片数据的意义
#     """
#     cur_path = os.path.dirname(__file__)
#     test_img_path = os.path.join(cur_path, "imgs_src", "BS13107_105-64-280-64-275-104-100-104.jpg")
#     test_image = Image.open(test_img_path)
#     print('test img size: ', test_image.size)
#     draw = ImageDraw.Draw(test_image)
#     draw.line((105, 64, 280, 64), fill=128)
#     draw.line((275, 104, 110, 104), fill=256)
#     test_image.show()


def generate_coor():
    """
    左上坐标：x,y分别取四个坐标中x,y最小值
    右下坐标：x,y分别取四个坐标中x,y最大值
    :return: 左上右下的坐标
    """
    cur_path = os.path.dirname(__file__)
    imgs_src_dir = os.path.join(cur_path, "imgs_src")
    images_dir = os.path.join(cur_path, 'transformed_imgs', 'JPEGImages')
    imgs_name_list = os.listdir(imgs_src_dir)
    for image_name in imgs_name_list:
        if not '_' in image_name:
            print('ignore the file, because file name: {} is invalid' % image_name)
            continue
        label_splits = image_name.split('_')
        if len(label_splits) > 2:
            print('ignore the file,because the file name {} has more than one \'_\'' % image_name)
            continue
        label = label_splits[0]
        other = label_splits[1]


if __name__ == "__main__":
    generate_coor()
