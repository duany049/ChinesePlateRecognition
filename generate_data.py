# coding=utf-8

import argparse
from PIL import Image, ImageFont, ImageDraw
import numpy as np
import cv2
import os
import utils
from constant import *


class PlateGenerator:
    def __init__(self, font_ch_path, font_en_path, bg_dir):
        self.font_ch = ImageFont.truetype(font_ch_path, 43, 0)
        self.font_en = ImageFont.truetype(font_en_path, 60, 0)
        self.board = np.asarray(Image.new("RGB", (226, 70), (255, 255, 255)))
        self.smudginess = cv2.imread('./data/bg/smudginess.jpg')
        self.bg = cv2.resize(cv2.imread('./data/bg/template.bmp'), (226, 70))

    def _add_smudginess(self, img, sum):
        # 看作用
        pass

    def _rot_img(self, img, angel, shape, max_angel):
        pass

    def _random_rot(self):
        pass

    def _random_bg(self):
        pass

    def _gen_char(self):
        pass

    def _add_gauss(self):
        pass

    def _add_noise(self):
        pass

    def _draw_chinese(self):
        pass

    def _draw_other_char(self):
        pass

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
        plate_img = cv2.bitwise_or(content_img, self.bg)
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
