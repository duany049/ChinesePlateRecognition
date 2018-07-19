# coding=utf-8
from .imdb import imdb
import os
from model.config import cfg
import pickle
import numpy as np
import scipy.sparse
import scipy.io as sio


class car_plate(imdb):
    """
    实现处理自己的数据集的接口的类
    """

    def __init__(self, image_set):
        """
        :param image_set: train or test
        """
        assert (image_set == 'train' or image_set == 'test'), 'invalid image_set: {0}'.format(image_set)
        name = 'car_plate_' + image_set
        imdb.__init__(self, name)
        self._image_set = image_set
        self._data_path = _get_default_path()
        self._classes = ('__backgroud__', 'plate')
        # 构成字典{'__background__':'0','plate':'1'}
        self._class_to_ind = dict(zip(self._classes, range(len(self._classes))))
        self._image_ext = '.jpg'
        # 读取train.txt，获取图片名称（该图片名称没有后缀.jpg）
        self._image_index = self._load_image_set_index()
        self._roidb_handler = self.gt_roidb
        self.config = {'cleanup': True,
                       'use_salt': True,
                       'top_k': 2000}

        assert os.path.exists(self._data_path), 'Image path: {} not exit'.format(self._data_path)

    def image_path_at(self, i):
        """
        :return: 根据image_set获取对应的数据集的path list
        """
        image_path = os.path.join(self._data_path, self._image_index[i], self._image_ext)
        assert os.path.exists(image_path), 'image path: {} not exists'.format(image_path)
        return image_path

    def gt_roidb(self):
        """
        Return the database of ground-truth regions of interest.
        This function loads/saves from/to a cache file to speed up future calls.
        :return:
        """
        cache_file = os.path.join(self.cache_path, self.name + '_gt_roidb.pkl')
        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as fid:
                try:
                    roidb = pickle.load(fid)
                except:
                    roidb = pickle.load(fid, encoding='bytes')
            print('{} gt db loaded from {}'.format(self.name, cache_file))
            return roidb

        gt_roidb = self._load_annotation()
        with open(cache_file, 'wb') as fid:
            pickle.dump(gt_roidb, fid, pickle.HIGHEST_PROTOCOL)
        print('wrote gt roidb to {}'.format(cache_file))
        return gt_roidb

    def _load_image_set_index(self):
        image_set_dir = os.path.join(self._data_path, self._image_set)
        image_names = os.listdir(image_set_dir)
        image_index = [name.split('.jpg')[0] for name in image_names]
        return image_index

    def competition_mode(self, on):
        if on:
            self.config['use_salt'] = False
            self.config['cleanup'] = False
        else:
            self.config['use_salt'] = True
            self.config['cleanup'] = True

    def _load_annotation(self):
        """
        读取图片集的gt信息的实现
        """
        gt_roidb = []
        info_file = os.path.join(self._data_path, self._image_set, 'infos.txt')
        with open(info_file, 'wb') as f:
            split_line = f.readline().strip().split(' ')
            while (split_line):
                # 我目前数据集都是单目标
                num_objs = 1
                boxes = np.zeros((num_objs, 4), dtype=np.uint16)
                gt_classes = np.zeros((num_objs,), dtype=np.int16)
                overlaps = np.zeros((num_objs, self.num_classes), dtype=np.float32)
                for i in range(num_objs):
                    x1 = float(split_line[1 + i * 4])
                    y1 = float(split_line[2 + i * 4])
                    x2 = float(split_line[3 + i * 4])
                    y2 = float(split_line[4 + i * 4])
                    cls = self._class_to_ind('plate')
                    boxes[i, :] = [x1, y1, x2, y2]
                    gt_classes[i] = cls
                    overlaps[i, cls] = 1.0

                overlaps = scipy.sparse.csc_matrix(overlaps)
                gt_roidb.append({'boxes': boxes, 'gt_classes': gt_classes, 'overlaps': overlaps, 'flipped': False})
                split_line = f.readline().strip().split(' ')
        return gt_roidb

    def _load_image_set_info(self):
        """
        todo 确认下返回的数据是否是从0增大？ 否则会出bug
        :return: 图片信息集
        """
        image_infos_set_file = os.path.join(self._data_path, self._image_set, 'infos.txt')
        assert os.path.exists(image_infos_set_file), 'path: {} is not exists'.format(image_set_file)
        with open(image_infos_set_file) as f:
            image_infos = [x.strip() for x in f.readlines()]
        return image_infos

    def _get_default_path(self):
        """
        Return the default path where car plate is expected to be installed.
        """
        return os.path.join(cfg.DATA_DIR, 'transfrom_format', 'transformed_imgs')
