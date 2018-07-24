# coding=utf-8
from .network import Network
import tensorflow as tf
import tensorflow.contrib.slim as slim


class Vgg16(Network):
    """
    使用vgg结构进行特征提取.
    """

    def __init__(self):
        Network.__init__(self)
        # 经过四次pooling,图片缩小了16倍
        self._feat_stride = [16, ]
        self._feat_compress = [1. / float(self._feat_stride[0]), ]
        self._scope = 'vgg_16'

    def _image_feature_extract(self, is_training, reuse=None):
        """
        定义图片特征提取网络.
        """
        # 统一scope用于区分不同的特征提取网络的参数
        with tf.variable_scope(self._scope, self._scope, reuse=reuse):
            # 参数从预训练模型中加载，只微调后面几层参数
            net = slim.repeat(self._image, 2, slim.conv2d, 64, [3, 3], trainable=False, scope='conv1')
            net = slim.max_pool2d(net, [2, 2], padding='SAME', scope='pool1')
            net = slim.repeat(net, 2, slim.conv2d, 128, [3, 3], trainable=False, scope='conv2')
            net = slim.max_pool2d(net, [2, 2], padding='SAME', scope='pool2')
            net = slim.repeat(net, 3, slim.conv2d, 256, [3, 3], trainable=is_training, scope='conv3')
            net = slim.max_pool2d(net, [2, 2], padding='SAME', scope='pool3')
            net = slim.repeat(net, 3    , slim.conv2d, 512, [3, 3], trainable=is_training, scope='conv4')
            net = slim.max_pool2d(net, [2, 2], padding='SAME', scope='pool4')
            net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], trainable=is_training, scope='conv5')
            # net = slim.max_pool2d(net, [2, 2], padding='SAME', scope='pool5')
        self._act_summaries.append(net)
        self._child_layers['head'] = net
        return net

    def _full_connect_layer(self, pool5, is_training, reuse=None):
        """
        定义全连接网络.
        """
        with tf.variable_scope(self._scope, self._scope, reuse=reuse):
            pool5_flatten = slim.flatten(pool5, scope='flatten')
            fc6 = slim.fully_connected(pool5_flatten, 4096, scope='fc6')
            if is_training:
                fc6 = slim.dropout(fc6, keep_prob=0.5, is_training=True, scope='dropout6')
            fc7 = slim.fully_connected(fc6, 4096, scope='fc7')
            if is_training:
                fc7 = slim.dropout(fc7, keep_prob=0.5, is_training=True, scope='dropout7')
        return fc7

    def get_variables_to_restore(self, variables, var_keep_dic):
        variables_to_restore = []

        for v in variables:
            # exclude the conv weights that are fc weights in vgg16
            if v.name == (self._scope + '/fc6/weights:0') or \
                    v.name == (self._scope + '/fc7/weights:0'):
                self._variables_to_transform[v.name] = v
                continue
            # exclude the first conv layer to swap RGB to BGR
            if v.name == (self._scope + '/conv1/conv1_1/weights:0'):
                self._variables_to_transform[v.name] = v
                continue
            if v.name.split(':')[0] in var_keep_dic:
                print('Variables restored: %s' % v.name)
                variables_to_restore.append(v)

        return variables_to_restore

    def transform_variables(self, sess, pretrained_model):
        """
        TODO 分别试试转换后的效果和不转换的效果
        """
        print("transform vgg16 layers")
        with tf.variable_scope('transform_vgg16'):
            with tf.device("/gpu:0"):
                # transform the vgg16 issue from conv weights to fc weights
                # transform RGB to BGR
                fc6_conv = tf.get_variable("fc6_conv", [7, 7, 512, 4096], trainable=False)
                fc7_conv = tf.get_variable("fc7_conv", [1, 1, 4096, 4096], trainable=False)
                conv1_rgb = tf.get_variable("conv1_rgb", [3, 3, 3, 64], trainable=False)
                restorer_variables = tf.train.Saver({self._scope + "/fc6/weights": fc6_conv,
                                                     self._scope + "/fc7/weights": fc7_conv,
                                                     self._scope + "/conv1/conv1_1/weights": conv1_rgb})
                restorer_variables.restore(sess, pretrained_model)

                sess.run(tf.assign(self._variables_to_transform[self._scope + '/fc6/weights:0'],
                                   tf.reshape(fc6_conv, self._variables_to_transform[
                                       self._scope + '/fc6/weights:0'].get_shape())))
                sess.run(tf.assign(self._variables_to_transform[self._scope + '/fc7/weights:0'],
                                   tf.reshape(fc7_conv, self._variables_to_transform[
                                       self._scope + '/fc7/weights:0'].get_shape())))
                sess.run(tf.assign(self._variables_to_transform[self._scope + '/conv1/conv1_1/weights:0'],
                                   tf.reverse(conv1_rgb, [2])))
