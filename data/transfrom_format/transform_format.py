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


def _generate_format_imgs():
    cur_path = os.path.dirname(__file__)
    imgs_src_dir = os.path.join(cur_path, "imgs_src")
    images_out_dir = os.path.join(cur_path, 'transformed_imgs', 'JPEGImages')
    imgs_name_list = os.listdir(imgs_src_dir)

    labels = []
    coors_list = []
    index = 0
    for image_name in imgs_name_list:
        label, coors = _generate_format_img(image_name)
        if label is None:
            print('ignore None result in generate format img')
            continue
        dest_name = ("%05d" % index) + '.jpg'
        index += 1
        img_src_all_path = os.path.join(imgs_src_dir, image_name)
        img_out_all_path = os.path.join(images_out_dir, dest_name)
        print('img out all path: ', img_out_all_path)
        Image.open(img_src_all_path).save(img_out_all_path)
        labels.append(label)
        coors_list.append(coors)
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
    return label, (x_start, y_start, x_end, y_end)


if __name__ == "__main__":
    _generate_format_imgs()
