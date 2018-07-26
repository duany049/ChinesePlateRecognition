# coding=utf-8

#from ...tools import _init_paths
import matplotlib.pyplot as plt
import numpy as np
import os.path as osp
import sys


import _init_paths_test
print(sys.path)
from lib.common.config import cfg
import argparse
import os
from lib.test.vgg16 import Vgg16
import random
import tensorflow as tf
from tensorflow.python import pywrap_tensorflow
import time
from lib.utils.timer import Timer
from my_utils import *

NETS = {'vgg16': ('vgg16_faster_rcnn_iter_70000.ckpt',), 'res101': ('res101_faster_rcnn_iter_110000.ckpt',)}
DATASETS = {'pascal_voc': ('voc_2007_trainval',), 'pascal_voc_0712': ('voc_2007_trainval+voc_2012_trainval',)}
CHARS = [u"京", u"沪", u"津", u"渝", u"冀", u"晋", u"蒙", u"辽", u"吉", u"黑", u"苏", u"浙", u"皖", u"闽", u"赣", u"鲁",
         u"豫", u"鄂", u"湘", u"粤", u"桂", u"琼", u"川", u"贵", u"云", u"藏", u"陕", u"甘", u"青", u"宁", u"新",
         u"0", u"1", u"2", u"3", u"4", u"5", u"6", u"7", u"8", u"9",
         u"A", u"B", u"C", u"D", u"E", u"F", u"G", u"H", u"J", u"K", u"L", u"M", u"N", u"P", u"Q", u"R", u"S", u"T",
         u"U", u"V", u"W", u"X", u"Y", u"Z"]

NUM_CHARS = len(CHARS)


def parse_args():
    """解析输入参数"""
    parser = argparse.ArgumentParser(description='test classify code')
    parser.add_argument('--net', dest='model_net', help='Model to use [vgg16 res101]',
                        choices=NETS.keys(), default='vgg16')
    parser.add_argument('--dataset', dest='dataset', help='Trained dataset [pascal_voc pascal_voc_0712]',
                        choices=DATASETS.keys(), default='pascal_voc_0712')
    args = parser.parse_args()
    return args


# 获取数据
class DataGain(object):
    def __init__(self, data_dir):
        self.images_path = []
        for root_path, sub_folder_name_list, file_name_list in os.walk(data_dir):
            self.images_path += [os.path.join(root_path, file_name) for file_name in file_name_list]
        random.shuffle(self.images_path)
        # TODO 检查labels正确性
        self.labels = [image_path[len(data_dir):].split('_')[0] for image_path in self.images_path]

    @property
    def size(self):
        return len(self.labels)

    def input_pipeline(self, batch_size, num_epochs=None):
        images_tensor = tf.convert_to_tensor(self.images_path, dtype=tf.string)
        labels_tensor = tf.convert_to_tensor(self.labels, dtype=tf.int64)
        # 将image_list ,label_list做一个slice处理,每次只生成一对数据，循环num_epochs次，如果为None就不限制循环次数
        image_path, label = tf.train.slice_input_producer([images_tensor, labels_tensor], num_epochs=num_epochs)
        image_content = tf.read_file(image_path)
        # image = tf.image.convert_image_dtype(tf.image.decode_jpeg(image_content, channels=1), tf.float32)
        # TODO 变成一纬的挺好的，我现在是为了还原当时的场景，以后可以改成单通道
        image = tf.image.convert_image_dtype(tf.image.decode_jpeg(image_content, channels=3), tf.float32)
        # TODO 我得定义一个size
        new_size = tf.constant([cfg.MY.WIDTH, cfg.MY.HEIGTH], dtype=tf.int32)
        image = tf.image.resize_images(image, new_size)
        print("image: %s label: %s" % (image, label))
        # 根据输入的元素，生成文件名队列，batch是每次出队的数目,capacity是总数,需要和启动文件名填充线程配合执行
        image_batch, label_batch = tf.train.shuffle_batch([image, label], batch_size=batch_size, capacity=50000,
                                                          min_after_dequeue=10000)
        return image_batch, label_batch


class SolverWrapper(object):
    def __init__(self, net, output_dir, tb_dir, model_name, pretrained_model=None):
        self.net = net

        self.model_name = model_name
        self.output_dir = output_dir
        self.tbdir = tb_dir
        self.pretrained_model = pretrained_model
        self.tbvaldir = tb_dir + '_val'
        if not os.path.exists(self.tbvaldir):
            os.makedirs(self.tbvaldir)

    def train_model(self, sess, max_iters):
        lr, train_op, global_step = self.construct_graph(sess)
        train_feeder = DataGain(data_dir='/data/CarPlate/plate')
        train_images, train_labels = train_feeder.input_pipeline(batch_size=cfg.MY.IMG_BATCH)
        saver = tf.train.Saver()
        # 返回最后一个保存的checkpoint文件名path，如果不存在就返回None
        ckpt = tf.train.latest_checkpoint(self.output_dir)
        # 如果存在模型则加载，如果不存在就当第一次进行训练，初始化
        if ckpt:
            saver.restore(sess, ckpt)
            print('restore from {0} checkpoint'.format(ckpt))
        else:
            self.initialize(sess)

        last_summary_time = time.time()
        timer = Timer()
        # 多线程协调器
        coord = tf.train.Coordinator()
        # 启动执行文件名队列填充的线程，之后计算单元才可以把数据读出来，否则文件名队列为空的，计算单元就会处于一直等待状态，导致系统阻塞
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        # train_writer = tf.summary.FileWriter(FLAGS.log_dir + '/train', sess.graph)
        # test_writer = tf.summary.FileWriter(FLAGS.log_dir + '/val')

        print(':::Training Start:::')
        try:
            while not coord.should_stop():
                timer.tic()
                # 执行多线程操作，进行入队出队得到batch
                train_batch_images, train_batch_labels = sess.run([train_images, train_labels])
                print('train_images_batch: {0} train_labels_batch: {1}'.format(train_batch_images.shape,
                                                                               train_batch_labels.shape))
                cls_targets = sparse_tuple_from(train_batch_labels)

                now = time.time()
                if now - last_summary_time > cfg.TRAIN.SUMMARY_INTERVAL:
                    loss_cls, summary, step, ctc_acc = self.net.train_step_with_summary(sess, train_batch_images,
                                                                                        cls_targets, train_op,
                                                                                        global_step)
                    last_summary_time = now
                    print('=====ctc acc: {}======'.format(ctc_acc))
                else:
                    # Compute the graph without summary
                    loss_cls, step = self.net.train_step(sess, train_batch_images, cls_targets, train_op, global_step)

                timer.toc()
                if step % 1 == 0:
                    print('iter: %d / %d, loss_cls: %.6f\n >>> lr: %f' % (step, max_iters, loss_cls, lr.eval()))
                    print('speed: {:.3f}s / iter'.format(timer.average_time))

                if step > max_iters:
                    break
                # if step % cfg.MY.EVAL_STEPS == 1:
                #     self.net.test
                if step % cfg.MY.SAVE_STEPS == 1:
                    print('Save the ckpt of step {0}'.format(step))
                    saver.save(sess, os.path.join(self.output_dir, self.model_name), global_step=global_step)

        except tf.errors.OutOfRangeError:
            print('==================Train Finished================')
            saver.save(sess, os.path.join(self.output_dir, self.model_name), global_step=global_step)
        finally:
            # 达到最大训练迭代数的时候请求关闭线程
            coord.request_stop()
        coord.join(threads)

    def initialize(self, sess):
        print('Loading initial model weights from {:s}'.format(pretrained_model))
        variables = tf.global_variables()
        # Initialize all variables first
        sess.run(tf.variables_initializer(variables, name='init'))
        var_keep_dic = self.get_variables_in_checkpoint_file(self.pretrained_model)
        # Get the variables to restore, ignoring the variables to fix
        variables_to_restore = self.net.get_variables_to_restore(variables, var_keep_dic)
        restorer = tf.train.Saver(variables_to_restore)
        restorer.restore(sess, self.pretrained_model)
        print('Loaded.')
        # Need to fix the variables before loading, so that the RGB weights are changed to BGR
        # For VGG16 it also changes the convolutional weights fc6 and fc7 to
        # fully connected weights
        self.net.transform_variables(sess, self.pretrained_model)
        print('Transform.')

    def construct_graph(self, sess):
        with sess.graph.as_default():
            layers = self.net.create_architecture('TRAIN', NUM_CHARS, tag='default')
            loss = layers['total_loss']
            lr = tf.Variable(cfg.TRAIN.LEARNING_RATE, trainable=False)
            global_step = tf.get_variable("step", [], initializer=tf.constant_initializer(0.0), trainable=False)
            self.optimizer = tf.train.MomentumOptimizer(lr, cfg.TRAIN.MOMENTUM)

            # Compute the gradients with regard to the loss
            gvs = self.optimizer.compute_gradients(loss)
            # Double the gradient of the bias if set
            if cfg.TRAIN.DOUBLE_BIAS:
                final_gvs = []
                with tf.variable_scope('Gradient_Mult') as scope:
                    for grad, var in gvs:
                        scale = 1.
                        if cfg.TRAIN.DOUBLE_BIAS and '/biases:' in var.name:
                            scale *= 2.
                        if not np.allclose(scale, 1.0):
                            grad = tf.multiply(grad, scale)
                        final_gvs.append((grad, var))
                train_op = self.optimizer.apply_gradients(final_gvs, global_step=global_step)
            else:
                train_op = self.optimizer.apply_gradients(gvs, global_step=global_step)

            # We will handle the snapshots ourselves
            self.saver = tf.train.Saver(max_to_keep=100000)
            # Write the train and validation information to tensorboard
            self.writer = tf.summary.FileWriter(self.tbdir, sess.graph)
            self.valwriter = tf.summary.FileWriter(self.tbvaldir)

            return lr, train_op, global_step

    def get_variables_in_checkpoint_file(self, file_name):
        try:
            reader = pywrap_tensorflow.NewCheckpointReader(file_name)
            var_to_shape_map = reader.get_variable_to_shape_map()
            return var_to_shape_map
        except Exception as e:  # pylint: disable=broad-except
            print(str(e))
            if "corrupted compressed block contents" in str(e):
                print("It's likely that your checkpoint file has been compressed "
                      "with SNAPPY.")


if __name__ == '__main__':
    net = Vgg16()
    # TODO 写vgg16权值预存目录
    pretrained_model = os.path.join('data/imagenet_weights/vgg_16.ckpt')
    model_name = 'test_car_plate'
    output_dir = 'output/vgg16/' + model_name
    tb_dir = 'tensorboard/vgg16/' + model_name + '/default'
    tfconfig = tf.ConfigProto(allow_soft_placement=True)
    tfconfig.gpu_options.allow_growth = True
    with tf.Session(config=tfconfig) as sess:
        sw = SolverWrapper(net, output_dir, tb_dir, model_name,
                           pretrained_model=pretrained_model)
        sw.train_model(sess, cfg.MY.MAX_ITERS)
