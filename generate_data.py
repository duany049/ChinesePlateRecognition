# coding=utf-8

import argparse
from PIL import Image, ImageFont, ImageDraw
import numpy as np
import cv2
import os
import utils
from constant import *
from utils import *
import random


class PlateGenerator:
    def __init__(self, font_ch_path, font_en_path, bg_dir):
        self.font_ch = ImageFont.truetype(font_ch_path, 43, 0)
        self.font_en = ImageFont.truetype(font_en_path, 60, 0)
        self.board = np.asarray(Image.new("RGB", (226, 70), (255, 255, 255)))
        self.smudginess = cv2.imread('./data/bg/smudginess.jpg')
        self.bg = cv2.resize(cv2.imread('./data/bg/template.bmp'), (226, 70))

    def _add_smudginess(self, img, smudginess):
        """
        :param img: src img
        :param smudginess: the src of smudginess
        :return: img that has smudginess
        """
        src_h, src_w = img.shape[:2]
        row_start = random_range(smudginess.shape[0] - src_h)
        col_start = random_range(smudginess.shape[1] - src_w)
        adder = smudginess[row_start:row_start + src_h, col_start:col_start + src_w]
        adder = cv2.resize(adder, (src_w, src_h))
        alpha = random.random() * 0.5
        beta = 1 - alpha
        fused_img = cv2.addWeighted(img, alpha, adder, beta, 0.0)
        return fused_img

    def _rot_img(self, img, angel, shape, max_angel):
        pass

    def _random_rot(self, img, scale, shape):
        pts1 = np.float32([[0, 0], [0, shape[0]], [shape[1], 0], [shape[1], shape[0]]])
        pts2 = np.float32([[random_range(scale), random_range(scale)],
                           [random_range(scale), shape[0] - random_range(scale)],
                           [shape[1] - random_range(scale), 0],
                           [shape[1] - random_range(scale), shape[0] - random_range(scale)]])
        matrix_transform = cv2.getPerspectiveTransform(pts1, pts2)
        warp_img = cv2.warpPerspective(img, matrix_transform, shape)
        return warp_img

    def _random_bg(self):
        pass

    def _gen_char(self):
        pass

    def _add_gauss(self, img, level):
        return cv2.blur(img, (level * 2 + 1, level * 2 + 1))

    def _add_noise_single_noise(self, img):


    def _add_noise(self, img):
        img[:, :, 0] = self._add_noise_single_noise(img[:, :, 0])
        img[:, :, 1] = self._add_noise_single_noise(img[:, :, 1])
        img[:, :, 2] = self._add_noise_single_noise(img[:, :, 2])
        return img

    def _draw_chinese(self, char):
        img = Image.new("RGB", (45, 70), (255, 255, 255))
        draw = ImageDraw.Draw(img)
        draw.text((0, 3), char, (0, 0, 0), font=self.font_ch)
        img = img.resize((23, 70))
        img_np = np.asarray(img)
        return img_np

    def _draw_other_char(self, char):
        img = Image.new('RGB', (23, 70), (255, 255, 255))
        draw = ImageDraw.Draw(img)
        draw.text((0, 2), char, (0, 0, 0), font=self.font_en)
        img_np = np.asarray(img)
        return img_np

    def _draw(self, plate_content):
        first_char_end = start_offset + char_width
        second_char_start = first_char_end + char_interval
        second_char_end = second_char_start + char_width
        self.board[:, start_offset:first_char_end] = self._draw_chinese(self.font_ch, plate_content[0])
        self.board[:, second_char_start:second_char_end] = self._draw_other_char(self.font_en, plate_content[1])
        for i in range(5):
            base = second_char_end + third_char_offset + i * char_width + i * char_interval
            self.board[:, base:base + char_width] = self._draw_other_char(self.font_en, plate_content[i + 2])
        return self.board

    def _gen_content(self):
        """
        :return: the content of plate.
        """
        plate_char_num = 7
        content = ""
        for index in range(plate_char_num):
            if index == 0:
                char_index = utils.random_range(province_num)
                content += plate_chars[char_index]
            elif index == 1:
                char_index = province_num + figure_num + utils.random_range(letter_num)
                content += plate_chars[char_index]
            else:
                char_index = province_num + utils.random_range(figure_num + letter_num)
                content += plate_chars[char_index]
        return content

    def _gen_plate(self, plate_content):
        """
        :param plate_content: the content of plate.
        :return: plate img
        """
        content_img = self._draw(plate_content)
        plate_img = cv2.bitwise_not(content_img)
        plate_img = cv2.bitwise_or(content_img, self.bg)
        # self._rot_img()
        plate_img = self._random_rot(plate_img)
        plate_img = self._add_smudginess(plate_img)
        # random_envirment
        plate_img = self._add_gauss(plate_img)
        plate_img = self._add_noise(plate_img)
        return plate_img

    def batch_gen_plate(self, num, out_path, image_size):
        if not os.path.exists(out_path):
            os.makedirs(out_path)
        for i in range(num):
            content = self._gen_content()
            img = self._gen_plate(content)
            img = cv2.resize(img, image_size)
            filename = os.path.join(out_path, str(i).zfill(5) + '_' + content + '.jpg')
            cv2.imwrite(filename, img)
            print("filename: %s" % filename)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--font_chinese_path', default='./font/platech.ttf')
    parser.add_argument('--font_english_path', default='./font/platechar.ttf')
    parser.add_argument('--width', default=120, type=int, help='the width of generated img')
    parser.add_argument('--height', default=32, type=int, help='the height of generated img')
    parser.add_argument('--bg_dir', default='./data/bg')
    parser.add_argument('--train_dir', default='./data/train')
    parser.add_argument('--test_dir', default='./data/test')
    parser.add_argument('--num', default=10000, type=int, help='num of generate img')
    return parser.parse_args()


def main(args):
    dataGenerator = PlateGenerator(args.font_chinese_path, args.font_english_path, args.bg_dir)
    dataGenerator.generate_data()


if __name__ == '__main__':
    main(parse_args())
