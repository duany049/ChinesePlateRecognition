# coding=utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.contrib.slim import losses
from tensorflow.contrib.slim import arg_scope
import tensorflow.contrib.rnn as rnn
from tensorflow.python.framework.sparse_tensor import SparseTensor
from tensorflow.python.framework.sparse_tensor import SparseTensorValue

import numpy as np

from layer_utils.snippets import generate_anchors_pre, generate_anchors_pre_tf
from layer_utils.proposal_layer import proposal_layer, proposal_layer_tf
from layer_utils.proposal_top_layer import proposal_top_layer, proposal_top_layer_tf
from layer_utils.anchor_target_layer import anchor_target_layer
from layer_utils.proposal_target_layer import proposal_target_layer
from utils.visualization import draw_bounding_boxes

from model.config import cfg
from my_utils import sparse_tuple_from
from my_utils import decode_sparse_tensor


class Network(object):
    def __init__(self):
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
        self._variables_to_transform = {}

    def _add_gt_image(self):
        # add back mean
        image = self._image + cfg.PIXEL_MEANS
        # BGR to RGB (opencv uses BGR)
        resized = tf.image.resize_bilinear(image, tf.to_int32(self._im_info[:2] / self._im_info[2]))
        self._gt_image = tf.reverse(resized, axis=[-1])

    def _add_gt_image_summary(self):
        # use a customized visualization function to visualize the boxes
        if self._gt_image is None:
            self._add_gt_image()
        image = tf.py_func(draw_bounding_boxes,
                           [self._gt_image, self._gt_boxes, self._im_info],
                           tf.float32, name="gt_boxes")

        return tf.summary.image('GROUND_TRUTH', image)

    def _add_act_summary(self, tensor):
        tf.summary.histogram('ACT/' + tensor.op.name + '/activations', tensor)
        tf.summary.scalar('ACT/' + tensor.op.name + '/zero_fraction',
                          tf.nn.zero_fraction(tensor))

    def _add_score_summary(self, key, tensor):
        # pass
        tf.summary.histogram('SCORE/' + tensor.op.name + '/' + key + '/scores', tensor)

    def _add_train_summary(self, var):
        tf.summary.histogram('TRAIN/' + var.op.name, var)

    def _reshape_layer(self, bottom, num_dim, name):
        input_shape = tf.shape(bottom)
        with tf.variable_scope(name) as scope:
            # change the channel to the caffe format
            to_caffe = tf.transpose(bottom, [0, 3, 1, 2])
            # then force it to have channel 2
            reshaped = tf.reshape(to_caffe,
                                  tf.concat(axis=0, values=[[1, num_dim, -1], [input_shape[2]]]))
            # then swap the channel back
            to_tf = tf.transpose(reshaped, [0, 2, 3, 1])
            return to_tf

    def _softmax_layer(self, bottom, name):
        if name.startswith('rpn_cls_prob_reshape'):
            input_shape = tf.shape(bottom)
            bottom_reshaped = tf.reshape(bottom, [-1, input_shape[-1]])
            reshaped_score = tf.nn.softmax(bottom_reshaped, name=name)
            return tf.reshape(reshaped_score, input_shape)
        return tf.nn.softmax(bottom, name=name)

    def _proposal_top_layer(self, rpn_cls_prob, rpn_bbox_pred, name):
        with tf.variable_scope(name) as scope:
            if cfg.USE_E2E_TF:
                rois, rpn_scores = proposal_top_layer_tf(
                    rpn_cls_prob,
                    rpn_bbox_pred,
                    self._im_info,
                    self._feat_stride,
                    self._anchors,
                    self._num_anchors
                )
            else:
                rois, rpn_scores = tf.py_func(proposal_top_layer,
                                              [rpn_cls_prob, rpn_bbox_pred, self._im_info,
                                               self._feat_stride, self._anchors, self._num_anchors],
                                              [tf.float32, tf.float32], name="proposal_top")

            rois.set_shape([cfg.TEST.RPN_TOP_N, 5])
            rpn_scores.set_shape([cfg.TEST.RPN_TOP_N, 1])

        return rois, rpn_scores

    def _proposal_layer(self, rpn_cls_prob, rpn_bbox_pred, name):
        with tf.variable_scope(name) as scope:
            if cfg.USE_E2E_TF:
                rois, rpn_scores = proposal_layer_tf(
                    rpn_cls_prob,
                    rpn_bbox_pred,
                    self._im_info,
                    self._mode,
                    self._feat_stride,
                    self._anchors,
                    self._num_anchors
                )
            else:
                rois, rpn_scores = tf.py_func(proposal_layer,
                                              [rpn_cls_prob, rpn_bbox_pred, self._im_info, self._mode,
                                               self._feat_stride, self._anchors, self._num_anchors],
                                              [tf.float32, tf.float32], name="proposal")

            rois.set_shape([None, 5])
            rpn_scores.set_shape([None, 1])

        return rois, rpn_scores

    # Only use it if you have roi_pooling op written in tf.image
    def _roi_pool_layer(self, bootom, rois, name):
        with tf.variable_scope(name) as scope:
            return tf.image.roi_pooling(bootom, rois,
                                        pooled_height=cfg.POOLING_SIZE,
                                        pooled_width=cfg.POOLING_SIZE,
                                        spatial_scale=1. / 16.)[0]

    def _crop_pool_layer(self, bottom, rois, name):
        with tf.variable_scope(name) as scope:
            batch_ids = tf.squeeze(tf.slice(rois, [0, 0], [-1, 1], name="batch_id"), [1])
            # Get the normalized coordinates of bounding boxes
            bottom_shape = tf.shape(bottom)
            height = (tf.to_float(bottom_shape[1]) - 1.) * np.float32(self._feat_stride[0])
            width = (tf.to_float(bottom_shape[2]) - 1.) * np.float32(self._feat_stride[0])
            x1 = tf.slice(rois, [0, 1], [-1, 1], name="x1") / width
            y1 = tf.slice(rois, [0, 2], [-1, 1], name="y1") / height
            x2 = tf.slice(rois, [0, 3], [-1, 1], name="x2") / width
            y2 = tf.slice(rois, [0, 4], [-1, 1], name="y2") / height
            # Won't be back-propagated to rois anyway, but to save time
            bboxes = tf.stop_gradient(tf.concat([y1, x1, y2, x2], axis=1))
            pre_pool_size = cfg.POOLING_SIZE * 2
            crops = tf.image.crop_and_resize(bottom, bboxes, tf.to_int32(batch_ids), [pre_pool_size, pre_pool_size],
                                             name="crops")

        return slim.max_pool2d(crops, [2, 2], padding='SAME')

    def _dropout_layer(self, bottom, name, ratio=0.5):
        return tf.nn.dropout(bottom, ratio, name=name)

    def _anchor_target_layer(self, rpn_cls_score, name):
        with tf.variable_scope(name) as scope:
            rpn_labels, rpn_bbox_targets, rpn_bbox_inside_weights, rpn_bbox_outside_weights = tf.py_func(
                anchor_target_layer,
                [rpn_cls_score, self._gt_boxes, self._im_info, self._feat_stride, self._anchors, self._num_anchors],
                [tf.float32, tf.float32, tf.float32, tf.float32],
                name="anchor_target")

            rpn_labels.set_shape([1, 1, None, None])
            rpn_bbox_targets.set_shape([1, None, None, self._num_anchors * 4])
            rpn_bbox_inside_weights.set_shape([1, None, None, self._num_anchors * 4])
            rpn_bbox_outside_weights.set_shape([1, None, None, self._num_anchors * 4])

            rpn_labels = tf.to_int32(rpn_labels, name="to_int32")
            self._anchor_targets['rpn_labels'] = rpn_labels
            self._anchor_targets['rpn_bbox_targets'] = rpn_bbox_targets
            self._anchor_targets['rpn_bbox_inside_weights'] = rpn_bbox_inside_weights
            self._anchor_targets['rpn_bbox_outside_weights'] = rpn_bbox_outside_weights

            self._score_summaries.update(self._anchor_targets)

        return rpn_labels

    def _proposal_target_layer(self, rois, roi_scores, name):
        with tf.variable_scope(name) as scope:
            rois, roi_scores, labels, bbox_targets, bbox_inside_weights, bbox_outside_weights = tf.py_func(
                proposal_target_layer,
                [rois, roi_scores, self._gt_boxes, self._num_classes],
                [tf.float32, tf.float32, tf.float32, tf.float32, tf.float32, tf.float32],
                name="proposal_target")

            rois.set_shape([cfg.TRAIN.BATCH_SIZE, 5])
            roi_scores.set_shape([cfg.TRAIN.BATCH_SIZE])
            # labels是k+1类别
            labels.set_shape([cfg.TRAIN.BATCH_SIZE, 1])
            bbox_targets.set_shape([cfg.TRAIN.BATCH_SIZE, self._num_classes * 4])
            bbox_inside_weights.set_shape([cfg.TRAIN.BATCH_SIZE, self._num_classes * 4])
            bbox_outside_weights.set_shape([cfg.TRAIN.BATCH_SIZE, self._num_classes * 4])

            self._proposal_targets['rois'] = rois
            self._proposal_targets['labels'] = tf.to_int32(labels, name="to_int32")
            self._proposal_targets['bbox_targets'] = bbox_targets
            self._proposal_targets['bbox_inside_weights'] = bbox_inside_weights
            self._proposal_targets['bbox_outside_weights'] = bbox_outside_weights

            self._score_summaries.update(self._proposal_targets)

            return rois, roi_scores

    def _anchor_component(self):
        with tf.variable_scope('ANCHOR_' + self._tag) as scope:
            # just to get the shape right
            height = tf.to_int32(tf.ceil(self._im_info[0] / np.float32(self._feat_stride[0])))
            width = tf.to_int32(tf.ceil(self._im_info[1] / np.float32(self._feat_stride[0])))
            if cfg.USE_E2E_TF:
                anchors, anchor_length = generate_anchors_pre_tf(
                    height,
                    width,
                    self._feat_stride,
                    self._anchor_scales,
                    self._anchor_ratios
                )
            else:
                anchors, anchor_length = tf.py_func(generate_anchors_pre,
                                                    [height, width,
                                                     self._feat_stride, self._anchor_scales, self._anchor_ratios],
                                                    [tf.float32, tf.int32], name="generate_anchors")
            anchors.set_shape([None, 4])
            anchor_length.set_shape([])
            self._anchors = anchors
            self._anchor_length = anchor_length

    def _build_network(self, is_training=True):
        # 是否使用truncated_normal_initializer初始化权重
        if cfg.TRAIN.TRUNCATED:
            initializer = tf.truncated_normal_initializer(mean=0.0, stddev=0.01)
            initializer_bbox = tf.truncated_normal_initializer(mean=0.0, stddev=0.001)
        else:
            initializer = tf.random_normal_initializer(mean=0.0, stddev=0.01)
            initializer_bbox = tf.random_normal_initializer(mean=0.0, stddev=0.001)

        net_conv = self._image_feature_extract(is_training)
        # with tf.variable_scope(self._scope, self._scope):
        # build the anchors for the image
        # self._anchor_component()
        # region proposal network
        # rois = self._region_proposal(net_conv, is_training, initializer)
        # region of interest pooling
        # if cfg.POOLING_MODE == 'crop':
        #     pool5 = self._crop_pool_layer(net_conv, rois, "pool5")
        # else:
        #     raise NotImplementedError
        fc7 = self._full_connect_layer(net_conv, is_training)
        with tf.variable_scope(self._scope, self._scope):
            # region classification
            # cls_prob, bbox_pred = self._region_classification(fc7, is_training,
            #                                                   initializer, initializer_bbox)
            cls_logits = self._region_classification(fc7, is_training,
                                                     initializer, initializer_bbox)

        self._score_summaries.update(self._predict_layers)

        # return rois, cls_logits, bbox_pred
        return cls_logits

    def _smooth_l1_loss(self, bbox_pred, bbox_targets, bbox_inside_weights, bbox_outside_weights, sigma=1.0, dim=[1]):
        sigma_2 = sigma ** 2
        box_diff = bbox_pred - bbox_targets
        in_box_diff = bbox_inside_weights * box_diff
        abs_in_box_diff = tf.abs(in_box_diff)
        smoothL1_sign = tf.stop_gradient(tf.to_float(tf.less(abs_in_box_diff, 1. / sigma_2)))
        in_loss_box = tf.pow(in_box_diff, 2) * (sigma_2 / 2.) * smoothL1_sign \
                      + (abs_in_box_diff - (0.5 / sigma_2)) * (1. - smoothL1_sign)
        out_loss_box = bbox_outside_weights * in_loss_box
        loss_box = tf.reduce_mean(tf.reduce_sum(
            out_loss_box,
            axis=dim
        ))
        return loss_box

    def _add_losses(self, sigma_rpn=3.0):
        with tf.variable_scope('LOSS_' + self._tag) as scope:
            # RPN, class loss
            # rpn_cls_score = tf.reshape(self._predict_layers['rpn_cls_score_reshape'], [-1, 2])
            # rpn_label = tf.reshape(self._anchor_targets['rpn_labels'], [-1])
            # rpn_select = tf.where(tf.not_equal(rpn_label, -1))
            # rpn_cls_score = tf.reshape(tf.gather(rpn_cls_score, rpn_select), [-1, 2])
            # rpn_label = tf.reshape(tf.gather(rpn_label, rpn_select), [-1])
            # rpn_cross_entropy = tf.reduce_mean(
            #     tf.nn.sparse_softmax_cross_entropy_with_logits(logits=rpn_cls_score, labels=rpn_label))

            # RPN, bbox loss
            # rpn_bbox_pred = self._predict_layers['rpn_bbox_pred']
            # rpn_bbox_targets = self._anchor_targets['rpn_bbox_targets']
            # rpn_bbox_inside_weights = self._anchor_targets['rpn_bbox_inside_weights']
            # rpn_bbox_outside_weights = self._anchor_targets['rpn_bbox_outside_weights']
            # rpn_loss_box = self._smooth_l1_loss(rpn_bbox_pred, rpn_bbox_targets, rpn_bbox_inside_weights,
            #                                     rpn_bbox_outside_weights, sigma=sigma_rpn, dim=[1, 2, 3])

            # RCNN, class loss
            # cls_score = self._predict_layers["cls_score"]
            # label = tf.reshape(self._proposal_targets["labels"], [-1])
            # cross_entropy = tf.reduce_mean(
            #     tf.nn.sparse_softmax_cross_entropy_with_logits(logits=cls_score, labels=label))

            # 使用ctc loss
            cls_logits = self._predict_layers["cls_logits"]
            # 暂时用不到这个,label是从其他数据计算过来的
            # cls_targets = self._predict_layers["cls_targets"]

            # cls_targets = sparse_tuple_from(label)
            ctc_loss = tf.nn.ctc_loss(self._cls_targets, cls_logits, self.seq_len)
            ctc_cost = tf.reduce_mean(ctc_loss)
            decoded, log_prob = tf.nn.ctc_beam_search_decoder(cls_logits, self.seq_len, merge_repeated=False)
            ctc_acc = tf.reduce_mean(tf.edit_distance(tf.cast(decoded[0], tf.int32), self._cls_targets))
            self._predict_layers['ctc_acc'] = ctc_acc
            self._predict_layers['decoded'] = decoded
            self._predict_layers['ctc_prob'] = log_prob

            # RCNN, bbox loss
            # bbox_pred = self._predict_layers['bbox_pred']
            # bbox_targets = self._proposal_targets['bbox_targets']
            # bbox_inside_weights = self._proposal_targets['bbox_inside_weights']
            # bbox_outside_weights = self._proposal_targets['bbox_outside_weights']
            # loss_box = self._smooth_l1_loss(bbox_pred, bbox_targets, bbox_inside_weights, bbox_outside_weights)

            # self._losses['cross_entropy'] = cross_entropy
            self._losses['ctc_cost'] = ctc_cost
            # self._losses['loss_box'] = loss_box
            # self._losses['rpn_cross_entropy'] = rpn_cross_entropy
            # self._losses['rpn_loss_box'] = rpn_loss_box

            # loss = cross_entropy + loss_box + rpn_cross_entropy + rpn_loss_box
            # loss = ctc_cost + loss_box + rpn_cross_entropy + rpn_loss_box
            # TODO 不知此regularization是否适用于ctc,古先注释掉
            # regularization_loss = tf.add_n(tf.losses.get_regularization_losses(), 'regu')
            # self._losses['total_loss'] = loss + regularization_loss
            # self._losses['total_loss'] = loss
            loss = ctc_loss
            self._losses['total_loss'] = loss
            self._event_summaries.update(self._losses)

        return loss

    def _region_proposal(self, net_conv, is_training, initializer):
        rpn = slim.conv2d(net_conv, cfg.RPN_CHANNELS, [3, 3], trainable=is_training, weights_initializer=initializer,
                          scope="rpn_conv/3x3")
        self._act_summaries.append(rpn)
        rpn_cls_score = slim.conv2d(rpn, self._num_anchors * 2, [1, 1], trainable=is_training,
                                    weights_initializer=initializer,
                                    padding='VALID', activation_fn=None, scope='rpn_cls_score')
        # change it so that the score has 2 as its channel size
        rpn_cls_score_reshape = self._reshape_layer(rpn_cls_score, 2, 'rpn_cls_score_reshape')
        rpn_cls_prob_reshape = self._softmax_layer(rpn_cls_score_reshape, "rpn_cls_prob_reshape")
        rpn_cls_pred = tf.argmax(tf.reshape(rpn_cls_score_reshape, [-1, 2]), axis=1, name="rpn_cls_pred")
        rpn_cls_prob = self._reshape_layer(rpn_cls_prob_reshape, self._num_anchors * 2, "rpn_cls_prob")
        rpn_bbox_pred = slim.conv2d(rpn, self._num_anchors * 4, [1, 1], trainable=is_training,
                                    weights_initializer=initializer,
                                    padding='VALID', activation_fn=None, scope='rpn_bbox_pred')
        if is_training:
            rois, roi_scores = self._proposal_layer(rpn_cls_prob, rpn_bbox_pred, "rois")
            rpn_labels = self._anchor_target_layer(rpn_cls_score, "anchor")
            # Try to have a deterministic order for the computing graph, for reproducibility
            with tf.control_dependencies([rpn_labels]):
                rois, _ = self._proposal_target_layer(rois, roi_scores, "rpn_rois")
        else:
            if cfg.TEST.MODE == 'nms':
                rois, _ = self._proposal_layer(rpn_cls_prob, rpn_bbox_pred, "rois")
            elif cfg.TEST.MODE == 'top':
                rois, _ = self._proposal_top_layer(rpn_cls_prob, rpn_bbox_pred, "rois")
            else:
                raise NotImplementedError

        self._predict_layers["rpn_cls_score"] = rpn_cls_score
        self._predict_layers["rpn_cls_score_reshape"] = rpn_cls_score_reshape
        self._predict_layers["rpn_cls_prob"] = rpn_cls_prob
        self._predict_layers["rpn_cls_pred"] = rpn_cls_pred
        self._predict_layers["rpn_bbox_pred"] = rpn_bbox_pred
        self._predict_layers["rois"] = rois

        return rois

    def _region_classification(self, fc7, is_training, initializer, initializer_bbox):
        # cls_score = slim.fully_connected(fc7, self._num_classes,
        #                                  weights_initializer=initializer,
        #                                  trainable=is_training,
        #                                  activation_fn=None, scope='cls_score')
        # cls_prob = self._softmax_layer(cls_score, "cls_prob")
        # cls_pred = tf.argmax(cls_score, axis=1, name="cls_pred")
        # 使用lstm代替softmax进行车牌识别,注意: 这儿训练图片使用cv读取,所以shape是(height, width, channel)
        # TODO 1,先使用fc7来试试效果,无论效果如何,都要用pool5直接试试,记得转换shape
        # TODO 2,NUM_HIDDEN 256和128都试试, NUM_LAYERS 1和2都试试

        # size为batch_size的以为数组,元素是每个待预测序列的长度
        self.seq_len = tf.placeholder(tf.int32, [None])
        # Here we use sparse_placeholder that will generate a
        # SparseTensor required by ctc_loss op.
        # targets = tf.sparse_placeholder(tf.int32)

        # 方式一: 使用fc7,shape为(batch_size, -1, 1), 最大时序为-1, feature为1
        # fc7_shape = tf.shape(fc7)
        fc7_shape = fc7.get_shape()
        # feature = tf.reshape(fc7, [fc7_shape[0], cfg.MY.MAX_TIMESTEP, 1])
        feature = tf.reshape(fc7, [fc7_shape[0], fc7_shape[1] * 4, -1])
        stack = rnn.MultiRNNCell([rnn.LSTMCell(cfg.MY.NUM_HIDDEN) for _ in range(cfg.MY.NUM_LAYERS)])
        outputs, _ = tf.nn.dynamic_rnn(stack, feature, self.seq_len, dtype=tf.float32)
        feature_shape = tf.shape(feature)
        batch_size, max_timesteps = feature_shape[0], feature_shape[1]
        # (batch_size * max_timesteps, num_hidden)
        outputs = tf.reshape(outputs, [-1, cfg.MY.NUM_HIDDEN])
        W = tf.Variable(tf.truncated_normal([cfg.MY.NUM_HIDDEN, cfg.MY.NUM_CLASSES], name='lstm_w'))
        b = tf.Variable(tf.constant(0., shape=[cfg.MY.NUM_CLASSES]), name='lstm_b')
        logits = tf.matmul(outputs, W) + b
        # Reshaping back to the original shape
        logits = tf.reshape(logits, [batch_size, max_timesteps, cfg.MY.NUM_CLASSES])
        # ctc使用下面这种形式(max_timesteps, batch_size, num_classes)
        logits = tf.transpose(logits, (1, 0, 2))
        self.ctc_decoded, ctc_cls_prob = tf.nn.ctc_beam_search_decoder(logits, self.seq_len, merge_repeated=False)

        # 方式二: 使用pool5,转换shape为(batch_size, width, height * channel)

        # bbox_pred = slim.fully_connected(fc7, self._num_classes * 4,
        #                                  weights_initializer=initializer_bbox,
        #                                  trainable=is_training,
        #                                  activation_fn=None, scope='bbox_pred')

        # self._predict_layers["cls_score"] = cls_score
        # self._predict_layers["cls_pred"] = cls_pred
        # self._predict_layers["cls_prob"] = cls_prob
        self._predict_layers["cls_logits"] = logits
        self._predict_layers['ctc_cls_prob'] = ctc_cls_prob
        # self._predict_layers["bbox_pred"] = bbox_pred
        self.seq_len_value = np.asarray([cfg.MY.MAX_TIMESTEP] * cfg.MY.IMG_BATCH, dtype=np.int32)
        self.seq_len_test_value = np.asarray([cfg.MY.MAX_TIMESTEP] * cfg.MY.IMG_BATCH, dtype=np.int32)
        # return cls_prob, bbox_pred
        # return logits, bbox_pred
        return logits

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

    def create_architecture(self, mode, num_classes, tag=None, ):
        self._image = tf.placeholder(tf.float32, shape=[1, None, None, 3])
        self._im_info = tf.placeholder(tf.float32, shape=[3])
        self._gt_boxes = tf.placeholder(tf.float32, shape=[None, 5])
        self._tag = tag

        self._data = tf.placeholder(tf.float32, shape=[cfg.MY.IMG_BATCH, cfg.MY.WIDTH, cfg.MY.HEIGTH, cfg.MY.CHANNELS])
        self._cls_targets = tf.sparse_placeholder(tf.int32)

        self._num_classes = num_classes
        self._mode = mode

        training = mode == 'TRAIN'
        testing = mode == 'TEST'

        assert tag != None

        # handle most of the regularizers here
        weights_regularizer = tf.contrib.layers.l2_regularizer(cfg.TRAIN.WEIGHT_DECAY)
        if cfg.TRAIN.BIAS_DECAY:
            biases_regularizer = weights_regularizer
        else:
            biases_regularizer = tf.no_regularizer

        # 定义当前作用域下这些网络层的参数，选择尽多类型的layer
        with arg_scope([slim.conv2d, slim.conv2d_in_plane, \
                        slim.conv2d_transpose, slim.separable_conv2d, slim.fully_connected],
                       weights_regularizer=weights_regularizer,
                       biases_regularizer=biases_regularizer,
                       biases_initializer=tf.constant_initializer(0.0)):
            # rois, cls_prob, bbox_pred = self._build_network(training)
            cls_logits = self._build_network(training)

        # layers_to_output = {'rois': rois}
        layers_to_output = {}

        for var in tf.trainable_variables():
            self._train_summaries.append(var)

        if testing:
            pass
            # stds = np.tile(np.array(cfg.TRAIN.BBOX_NORMALIZE_STDS), (self._num_classes))
            # means = np.tile(np.array(cfg.TRAIN.BBOX_NORMALIZE_MEANS), (self._num_classes))
            # self._predict_layers["bbox_pred"] *= stds
            # self._predict_layers["bbox_pred"] += means
        else:
            self._add_losses()
            layers_to_output.update(self._losses)

            val_summaries = []
            with tf.device("/cpu:0"):
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

        layers_to_output.update(self._predict_layers)

        return layers_to_output

    def get_variables_to_restore(self, variables, var_keep_dic):
        raise NotImplementedError

    def transform_variables(self, sess, pretrained_model):
        raise NotImplementedError

    # Extract the head feature maps, for example for vgg16 it is conv5_3
    # only useful during testing mode
    def extract_head(self, sess, image):
        feed_dict = {self._image: image}
        feat = sess.run(self._child_layers["head"], feed_dict=feed_dict)
        return feat

    # only useful during testing mode
    def test_image(self, sess, data, cls_targets):
        feed_dict = {self._data: data,
                     self._cls_targets: cls_targets,
                     self.seq_len: self.seq_len_test_value}

        cls_score, cls_prob, ctc_decoded, bbox_pred, rois = sess.run([self._predict_layers["cls_logits"],
                                                                      # self._predict_layers["cls_score"],
                                                                      # self._predict_layers['cls_prob'],
                                                                      self._predict_layers['ctc_cls_prob'],
                                                                      self.ctc_decoded[0],
                                                                      self._predict_layers['bbox_pred'],
                                                                      self._predict_layers['rois']],
                                                                     feed_dict=feed_dict)
        # print('dy test ctc_decoded shape: {} - value: {} - target: {}'
        #       .format(np.array(ctc_decoded).shape, ctc_decoded, cls_targets))
        # print('==============cls targets====================')
        # decode_sparse_tensor(cls_targets)
        # print('==============ctc decoded====================')
        decode_sparse_tensor(ctc_decoded)
        return cls_score, cls_prob, bbox_pred, rois

    def get_summary(self, sess, blobs):
        feed_dict = {self._image: blobs['data'], self._im_info: blobs['im_info'],
                     self._gt_boxes: blobs['gt_boxes'], self.seq_len: self.seq_len_value}
        summary = sess.run(self._summary_op_val, feed_dict=feed_dict)

        return summary

    def print_predict(self, dd, label):
        detected_list = decode_sparse_tensor(dd)
        original_list = decode_sparse_tensor(label)
        true_numer = 0
        if len(original_list) != len(detected_list):
            print("len(original_list)", len(original_list), "len(detected_list)", len(detected_list),
                  " test and detect length desn't match")
            return
        for idx, number in enumerate(original_list):
            detect_number = detected_list[idx]
            hit = (number == detect_number)
            print(hit, number, "(", len(number), ") <-------> ", detect_number, "(", len(detect_number), ")")
            if hit:
                true_numer = true_numer + 1
        print("Test Accuracy:", true_numer * 1.0 / len(original_list))

    def train_step_with_summary(self, sess, data, cls_targets, train_op, global_step):
        feed_dict = {self._data: data, self._cls_targets: cls_targets,
                     self.seq_len: self.seq_len_value}
        loss_cls, _, step, ctc_acc, dd, ctc_prob = sess.run([self._losses['ctc_cost'],
                                                   # self._summary_op,
                                                   train_op,
                                                   global_step,
                                                   self._predict_layers['ctc_acc'],
                                                   self._predict_layers['decoded'][0],
                                                   self._predict_layers['ctc_prob']],
                                                   feed_dict=feed_dict)
        self.print_predict(dd, cls_targets)
        return loss_cls, step, ctc_acc

    def train_step(self, sess, data, cls_targets, train_op, global_step):
        feed_dict = {self._data: data, self._cls_targets: cls_targets,
                     self.seq_len: self.seq_len_value}
        loss_cls, _, step = sess.run([self._losses['ctc_cost'], train_op, global_step], feed_dict=feed_dict)
        return loss_cls, step

    # def train_step(self, sess, blobs, train_op):
    #     feed_dict = {self._image: blobs['data'], self._im_info: blobs['im_info'],
    #                  self._gt_boxes: blobs['gt_boxes'],
    #                  self.seq_len: self.seq_len_value}
    #     rpn_loss_cls, rpn_loss_box, loss_cls, loss_box, loss, _ = sess.run([self._losses["rpn_cross_entropy"],
    #                                                                         self._losses['rpn_loss_box'],
    #                                                                         # self._losses['cross_entropy'],
    #                                                                         self._losses['ctc_cost'],
    #                                                                         self._losses['loss_box'],
    #                                                                         self._losses['total_loss'],
    #                                                                         train_op],
    #                                                                        # self._predict_layers['ctc_acc']],
    #                                                                        feed_dict=feed_dict)
    #     return rpn_loss_cls, rpn_loss_box, loss_cls, loss_box, loss

    # def train_step_with_summary(self, sess, blobs, train_op):
    #     feed_dict = {self._image: blobs['data'], self._im_info: blobs['im_info'],
    #                  self._gt_boxes: blobs['gt_boxes'],
    #                  self.seq_len: self.seq_len_value}
    #     rpn_loss_cls, rpn_loss_box, loss_cls, loss_box, loss, summary, _ = sess.run(
    #         [self._losses["rpn_cross_entropy"],
    #          self._losses['rpn_loss_box'],
    #          # self._losses['cross_entropy'],
    #          self._losses['ctc_cost'],
    #          self._losses['loss_box'],
    #          self._losses['total_loss'],
    #          self._summary_op,
    #          train_op],
    #         # self._predict_layers['ctc_acc']],
    #         feed_dict=feed_dict)
    #     return rpn_loss_cls, rpn_loss_box, loss_cls, loss_box, loss, summary

    def train_step_no_return(self, sess, blobs, train_op):
        feed_dict = {self._image: blobs['data'], self._im_info: blobs['im_info'],
                     self._gt_boxes: blobs['gt_boxes']}
        sess.run([train_op], feed_dict=feed_dict)
