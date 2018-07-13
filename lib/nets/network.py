# coding=utf-8
from lib.common.config import cfg
import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.contrib.slim import arg_scope
import numpy as np


class Network(object):
    """
    所有网络结构的父类,实现了rpn和fast rcnn公共接口
    """

    def __init__(self):
        # 预测结果层
        self._predict_layers = {}
        self._losses = {}
        self._anchor_targets = {}
        self._proposal_targets = {}
        self._child_layers = {}
        self._gt_image = None
        self._act_summaries = []
        self._score_summaries = {}
        self._train_summaries = []
        self._event_summaries = {}
        self._variables_to_fix = {}

    def _image_feature_extract(self, is_training, reuse=None):
        """
        定义图片特征提取网络.
        :param is_training:
        :param reuse:
        :return:
        """
        raise NotImplementedError

    def _full_connect_layer(self, pool5, is_training, reuse=None):
        """
        定义全连接网络.
        :param pool5:
        :param is_training:
        :param reuse:
        :return:
        """
        raise NotImplementedError

    # def get_variables_to_restore(self, variables, var_keep_dic):
    #     raise NotImplementedError

    def transform_variables(self, sess, pretrained_model):
        raise NotImplementedError

    def _anchor_component(self):
        pass

    def _softmax_layer(self, bottom, name):
        if name.startswith('rpn_cls_prob_reshape'):
            input_shape = tf.shape(bottom)
            bottom_reshaped = tf.reshape(bottom, [-1, input_shape[-1]])
            reshaped_score = tf.nn.softmax(bottom_reshaped, name=name)
        return tf.nn.softmax(bottom, name=name)

    def _add_losses(self, sigma_rpn=3.0):
        with tf.variable_scope('LOSS_' + self._tag):
            cls_score = self._predict_layers["cls_score"]
            # todo 情况特殊直接传入label
            label = self._label
            cross_entropy = tf.reduce_mean(
                tf.nn.sparse_softmax_cross_entropy_with_logits(logits=cls_score, labels=label))
            regularization_loss = tf.add_n(tf.losses.get_regularization_losses(), 'regu')
            self._losses['total_loss'] = cross_entropy + regularization_loss
            self._event_summaries.update(self._losses)
        return cross_entropy

        # with tf.variable_scope('LOSS_' + self._tag) as scope:
        #     # RPN, class loss
        #     rpn_cls_score = tf.reshape(self._predict_layers['rpn_cls_score_reshape'], [-1, 2])
        #     rpn_label = tf.reshape(self._anchor_targets['rpn_labels'], [-1])
        #     rpn_select = tf.where(tf.not_equal(rpn_label, -1))
        #     rpn_cls_score = tf.reshape(tf.gather(rpn_cls_score, rpn_select), [-1, 2])
        #     rpn_label = tf.reshape(tf.gather(rpn_label, rpn_select), [-1])
        #     rpn_cross_entropy = tf.reduce_mean(
        #         tf.nn.sparse_softmax_cross_entropy_with_logits(logits=rpn_cls_score, labels=rpn_label))
        #
        #     # RPN, bbox loss
        #     rpn_bbox_pred = self._predict_layers['rpn_bbox_pred']
        #     rpn_bbox_targets = self._anchor_targets['rpn_bbox_targets']
        #     rpn_bbox_inside_weights = self._anchor_targets['rpn_bbox_inside_weights']
        #     rpn_bbox_outside_weights = self._anchor_targets['rpn_bbox_outside_weights']
        #     rpn_loss_box = self._smooth_l1_loss(rpn_bbox_pred, rpn_bbox_targets, rpn_bbox_inside_weights,
        #                                         rpn_bbox_outside_weights, sigma=sigma_rpn, dim=[1, 2, 3])
        #
        #     # RCNN, class loss
        #     cls_score = self._predict_layers["cls_score"]
        #     label = tf.reshape(self._proposal_targets["labels"], [-1])
        #     cross_entropy = tf.reduce_mean(
        #         tf.nn.sparse_softmax_cross_entropy_with_logits(logits=cls_score, labels=label))
        #
        #     # RCNN, bbox loss
        #     bbox_pred = self._predict_layers['bbox_pred']
        #     bbox_targets = self._proposal_targets['bbox_targets']
        #     bbox_inside_weights = self._proposal_targets['bbox_inside_weights']
        #     bbox_outside_weights = self._proposal_targets['bbox_outside_weights']
        #     loss_box = self._smooth_l1_loss(bbox_pred, bbox_targets, bbox_inside_weights, bbox_outside_weights)
        #
        #     self._losses['cross_entropy'] = cross_entropy
        #     self._losses['loss_box'] = loss_box
        #     self._losses['rpn_cross_entropy'] = rpn_cross_entropy
        #     self._losses['rpn_loss_box'] = rpn_loss_box
        #
        #     loss = cross_entropy + loss_box + rpn_cross_entropy + rpn_loss_box
        #     regularization_loss = tf.add_n(tf.losses.get_regularization_losses(), 'regu')
        #     self._losses['total_loss'] = loss + regularization_loss
        #
        #     self._event_summaries.update(self._losses)
        #
        # return loss

    def _region_classification(self, fc7, is_training, initializer, initializer_bbox):
        # todo 暂时先实现了分类，即将实现回归检测框
        cls_score = slim.fully_connected(fc7, self._num_classes,
                                         weights_initializer=initializer,
                                         trainable=is_training,
                                         activation_fn=None, scope='cls_score')
        cls_prob = self._softmax_layer(cls_score, 'cls_prob')
        cls_pred = tf.argmax(cls_prob, axis=1, name='cls_pred')
        self._predict_layers['cls_score'] = cls_score
        self._predict_layers['cls_prob'] = cls_prob
        self._predict_layers['cls_pred'] = cls_pred
        # todo return bbox_pred
        _ = 0

        return cls_prob, _

    def _build_network(self, is_training=True):
        # 是否使用truncated_normal_initializer初始化权重
        if cfg.TRAIN.TRUNCATED:
            initializer = tf.truncated_normal_initializer(mean=0.0, stddev=0.01)
            initializer_bbox = tf.truncated_normal_initializer(mean=0.0, stddev=0.001)
        else:
            initializer = tf.random_normal_initializer(mean=0.0, stddev=0.01)
            initializer_bbox = tf.random_normal_initializer(mean=0.0, stddev=0.01)

        conv_net = self._image_feature_extract(is_training)
        with tf.variable_scope(self._scope, self._scope):
            # generate anchor
            self._anchor_component()
            # todo 待实现rpn

        # fc7 = self._head_of_classify(pool5, is_training)
        # todo 暂时先实现分类,等待实现检测
        fc7 = self._full_connect_layer(conv_net, is_training)
        with tf.variable_scope(self._scope, self._scope):
            cls_prob, _ = self._region_classification(fc7, is_training, initializer, initializer_bbox)

        self._score_summaries.update(self._predict_layers)
        return _, cls_prob, _

    def create_architecture(self, mode, num_classes, tag=None,
                            anchor_scales=(8, 16, 32), anchor_ratios=(0.5, 1, 2)):
        self._image = tf.placeholder(tf.float32, shape=[1, None, None, 3])
        self._im_info = tf.placeholder(tf.float32, shape=[3])
        self._gt_boxes = tf.placeholder(tf.float32, shape=[None, 5])
        self._tag = tag

        self._num_classes = num_classes
        # TRAIN or TEST
        self._mode = mode
        self._anchor_scales = anchor_scales
        self._num_scales = len(anchor_scales)
        self._anchor_ratios = anchor_ratios
        self._num_ratios = len(anchor_ratios)
        self._num_anchors = self._num_scales * self._num_ratios

        training = mode == 'TRAIN'
        testing = mode == 'TEST'

        assert tag != None

        weights_regularizer = tf.contrib.layer.l2_regularizer(cfg.TRAIN.WEIGHT_DECAY)
        if cfg.TRAIN.BIAS_DECAY:
            biases_regularizer = weights_regularizer
        else:
            biases_regularizer = tf.no_regularizer

        # 定义当前作用域下这些网络层的参数
        with arg_scope([slim.conv2d, slim.conv2d_in_plane, \
                        slim.conv2d_transpose, slim.separable_conv2d, slim.fully_connected],
                       weights_regularizer=weights_regularizer,
                       biases_regularizer=biases_regularizer,
                       biases_initializer=tf.constant_initializer(0.0)):
            rois, cls_prob, bbox_pred = self._build_network(training)

        layers_to_output = {'rois': rois}

        for var in tf.trainable_variables():
            self._train_summaries.append(var)

        if testing:
            stds = np.tile(np.array(cfg.TRAIN.BBOX_NORMALIZE_STDS), (self._num_classes))
            means = np.tile(np.array(cfg.TRAIN.BBOX_NORMALIZE_MEANS), (self._num_classes))
            self._predict_layers["bbox_pred"] *= stds
            self._predict_layers["bbox_pred"] += means
        else:
            self._add_losses()
            layers_to_output.update(self._losses)

            val_summaries = []
            with tf.device("/gpu:0"):
                val_summaries.append(self._add_gt_image_summary())
                for key, var in self._event_summaries.items():
                    val_summaries.append(tf.summary.scalar(key, var))
                for key, var in self._score_summaries.items():
                    self._add_score_summary(key, var)
                for var in self._act_summaries:
                    self._add_act_summary(var)
                for var in self._train_summaries:
                    self._add_train_summary(var)

            self._summary_op = tf.summary.merge_all()
            self._summary_op_val = tf.summary.merge(val_summaries)

        layers_to_output.update(self._predictions)

        return layers_to_output

    def test_image(self, sess, image, im_info):
        feed_dict = {self._image: image,
                     self._im_info: im_info}
        # todo 实现目标检测
        class_score = sess.run([self._predict_layers['cls_score']], feed_dict=feed_dict)
        return class_score
