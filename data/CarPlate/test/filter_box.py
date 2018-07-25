# coding=utf-8
import os
from PIL import Image, ImageDraw
import numpy as np
import cv2
import time

check_width = 2
check_height = 4
standard_rect = []
standard_line_rect = []
standard_element_rect = [255, 255, 255]
for w in range(check_width):
    standard_line_rect.append(standard_element_rect)
for h in range(check_height):
    standard_rect.append(standard_line_rect)

standard_sum = 255 * 2 * 4 * 3


def _has_pure_white_box(img_src, coors):
    """
    取(x1,y1)和(x4,y4)的中间点作为中心,然后左右各7像素,上下各十像素的矩形区域
    遍历查看此区域内有没有这种直线:BGR值为([225,225,225]),并且宽2像素,高至少4像素
    如果存在则认为是有白边框的图
    """
    img = cv2.imread(img_src)
    x1, y1, x2, y2, x3, y3, x4, y4 = coors
    left_center_y = int((y4 - y1) / 2 + y1)
    left_center_x = int((x4 - x1) / 2 + x1)
    # print('x1: {} - y1: {} - left_center_x: {} - left_center_y: {}'.format(x1, y1, left_center_x, left_center_y))
    horizontal = 10
    vertical = 14
    rect = img[left_center_y - vertical:left_center_y + vertical,
           left_center_x - horizontal: left_center_x + horizontal, :]
    print(_similarity_degree(img[left_center_y - 2:left_center_y + 2, left_center_x - 2:left_center_x, :]))

    img_img = Image.open(img_src)
    draw = ImageDraw.Draw(img_img)
    # draw.rectangle((x1,y1, x4,y4), fill=128)
    draw.rectangle((left_center_x - 2, left_center_y - 2, left_center_x, left_center_y + 2), fill=128)
    draw.rectangle((left_center_x - horizontal,left_center_y - vertical, left_center_x + horizontal, left_center_y + vertical), fill=128)
    img_img.show()
    # print(img[left_center_y - 2:left_center_y + 2, left_center_x - 2:left_center_x, :])
    min_error = -1
    for w in range(rect.shape[1] - 1):
        for h in range(rect.shape[0] - 3):
            check_rect = rect[h: h + check_height, w:w + check_width, :]
            error_value = _similarity_degree(check_rect)
            if min_error == -1:
                min_error = error_value
            elif min_error > error_value:
                min_error = error_value
            if error_value < 0.042:
                # w_start = left_center_x - horizontal + w
                # w_end = left_center_x - horizontal + w + check_width
                # h_start = left_center_y - vertical + h
                # h_end = left_center_y - vertical + h + check_height
                # img_img.show()
                return True, min_error
    return False, min_error


def _similarity_degree(check_rect):
    standard_rect_nd = np.asarray(standard_rect)
    check_rect_nd = np.asarray(check_rect)
    differ_rect = standard_rect_nd - check_rect_nd
    diff_sum = np.sum(differ_rect)
    error_value = diff_sum / standard_sum
    return error_value


if __name__ == '__main__':
    cur_path = os.path.dirname(__file__)
    imgs_src_dir = os.path.join(cur_path, 'src')
    images_out_dir = os.path.join(cur_path, 'filter_box_imgs')
    imgs_name_list = os.listdir(imgs_src_dir)

    # 如果没有白边框就存入filter_box_imgs目录下
    start_time = time.time()
    for image_name in imgs_name_list:
        # if not image_name == '川A228DP_66-58-166-77-166-120-66-101.jpg':
        #     continue

        if not '_' in image_name:
            print('ignore the file, because file name: {%s} is invalid' % image_name)
            continue
        label_splits = image_name.split('_')
        if len(label_splits) > 2:
            print('ignore the file,because the file name {} has more than one \'_\'' % image_name)
            continue
        coors_str = label_splits[1]
        coors = coors_str.split('-')
        if len(coors) != 8:
            print('ignore next coor action, because the coor: {0} is invalid'.format(coors))
            continue
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
            continue
        y4 = int(y4[0])

        img_path = os.path.join(imgs_src_dir, image_name)
        coors = (x1, y1, x2, y2, x3, y3, x4, y4)
        has_box, min_error = _has_pure_white_box(img_path, coors)
        if has_box:
            print('ignore: {}, because of has box - min error: {}'.format(image_name, min_error))
            continue
        else:
            print('consider: {} is useful img - min error: {}'.format(image_name, min_error))

        out_path = os.path.join(images_out_dir, image_name)
        Image.open(img_path).save(out_path)

    print('use time: ', time.time() - start_time)
