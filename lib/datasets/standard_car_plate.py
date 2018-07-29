# coding=utf-8
from datasets.imdb import imdb
import os
from model.config import cfg
import pickle
import numpy as np
import scipy.sparse
import scipy.io as sio
import xml.etree.ElementTree as ET


class standard_car_plate(imdb):
    """
    实现处理自己的标准数据集的接口的类
    """

    def __init__(self, image_set):
        """
        :param image_set: train or test
        """
        assert (image_set == 'train' or image_set == 'test'), 'invalid image_set: {0}'.format(image_set)
        name = 'standard_car_plate_' + image_set
        imdb.__init__(self, name)
        self._image_set = image_set
        self._data_path = self._get_default_path()
        # TODO 使用车牌的label
        self._classes = ('__backgroud__', 'plate')
        # 构成字典{'__background__':'0','plate':'1'}
        self._class_to_ind = dict(zip(self._classes, range(len(self._classes))))
        self._image_ext = '.jpg'
        # 读取Main中对应数据集的txt文件，获取图片名称（该图片名称没有后缀.jpg）
        self._image_index = self._load_image_set_names()
        self._roidb_handler = self.gt_roidb
        self.config = {'cleanup': True,
                       'use_salt': True,
                       'top_k': 2000}

        assert os.path.exists(self._data_path), 'Image path: {} not exit'.format(self._data_path)

    def image_path_at(self, index):
        """
        Construct an image path from the image's "index" identifier.
        """
        return self.image_path_from_index(self._image_index[index])

    def image_path_from_index(self, name):
        """
        Construct an image path from the image's "index" identifier.
        """
        image_path = os.path.join(self._data_path, 'JPEGImages',
                                  name + self._image_ext)
        assert os.path.exists(image_path), \
            'Path does not exist: {}'.format(image_path)
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

        gt_roidb = [self._load_annotation(name) for name in self._image_index]
        with open(cache_file, 'wb') as fid:
            pickle.dump(gt_roidb, fid, pickle.HIGHEST_PROTOCOL)
        print('wrote gt roidb to {}'.format(cache_file))
        return gt_roidb

    def competition_mode(self, on):
        if on:
            self.config['use_salt'] = False
            self.config['cleanup'] = False
        else:
            self.config['use_salt'] = True
            self.config['cleanup'] = True

    def _load_annotation(self, name):
        """
        Load img and box info from xml file
        """
        filename = os.path.join(self._data_path, 'Annotations', name + '.xml')
        tree = ET.parse(filename)
        objs = tree.findall('object')
        num_objs = len(objs)

        boxes = np.zeros((num_objs, 4), dtype=np.uint16)
        gt_classes = np.zeros((num_objs), dtype=np.int32)
        overlaps = np.zeros((num_objs, self.num_classes), dtype=np.float32)
        # "Seg" area for pascal is just the box area
        seg_areas = np.zeros((num_objs), dtype=np.float32)
        # TODO 实现labels
        labels = np.zeros((num_objs))

        # Load object bounding boxes into a data frame.
        for ix, obj in enumerate(objs):
            bbox = obj.find('bndbox')
            # Make pixel indexes 0-based
            x1 = float(bbox.find('xmin').text)
            y1 = float(bbox.find('ymin').text)
            x2 = float(bbox.find('xmax').text)
            y2 = float(bbox.find('ymax').text)
            cls = self._class_to_ind[obj.find('name').text.lower().strip()]
            boxes[ix, :] = [x1, y1, x2, y2]
            gt_classes[ix] = cls
            overlaps[ix, cls] = 1.0
            seg_areas[ix] = (x2 - x1 + 1) * (y2 - y1 + 1)


        overlaps = scipy.sparse.csr_matrix(overlaps)
        return {'boxes': boxes,
                'gt_classes': gt_classes,
                'gt_overlaps': overlaps,
                'flipped': False,
                'seg_areas': seg_areas}

    def _load_image_set_names(self):
        """
        :return: 图片信息集
        """
        image_set_file = os.path.join(self._data_path, 'ImageSets', 'Main', self._image_set + '.txt')
        assert os.path.exists(image_set_file), 'path: {} is not exists'.format(image_set_file)
        with open(image_set_file) as f:
            image_names = [x.strip() for x in f.readlines()]
        return image_names

    def _get_default_path(self):
        """
        Return the default path where car plate is expected to be installed.
        """
        return os.path.join(cfg.DATA_DIR, 'CarPlate', 'standard_data')


if __name__=='__main__':
    # 执行test部分
    standard_car_plate = standard_car_plate('train')
    roidb = standard_car_plate.gt_roidb()
    print(roidb)
