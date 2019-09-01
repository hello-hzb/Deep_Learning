# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import scipy.io as sio
import datetime

data_dir = '/home/ubuntu/my_file/hyper_sae/hyper_data/'

'''
##############################################
# 读取高光谱图像数据
load_fn = '../hyper_data/plane.mat'
# load_data = sio.loadmat(load_fn)
load_matrix = load_data['y']   # 100*100*126
facet = load_matrix.reshape(10000, 126)
facet = np.array(facet, dtype=np.float32)
trX = facet / facet.max()
trRef = trX    # float32
##############################################
'''


def read_tif(dir_filename):
    print("Read tif image! File name is", dir_filename)
    return


def write_tif(dir_filename):
    print("Write tif image! File name is ", dir_filename)
    return


def read_excel(dir_filename):  # 未完成，后续再写
    print ('Read excel! File name is ', dir_filename)

    print("It is ")
    return


def read_mat(dir_filename):
    print ("Read mat! File name is", dir_filename)
    load_data = sio.loadmat(dir_filename)
    print ("It is done!")
    return load_data


def write_mat(dir_filename, data1, data2, data3):
    print ("Write mat! File name is", dir_filename)
    # sio.savemat(dir_filename, {'array': data})
    sio.savemat(dir_filename, {'hidden_size': data1, 'time': data2, "num_epoch": data3})  # 同理，只是存入了两个不同的变量供使用
    print ("It is done!")
    return


def write_txt(dir_filename, data):
    print ("Write txt! File name is", dir_filename)
    np.savetxt(dir_filename, data)
    print ("It is done!")
    return


def read_txt(dir_filename):
    print ("Read txt! File name is ", dir)
    load_data = np.loadtxt(dir_filename)
    print ("It is done!")
    return load_data


def normalize_data(input_data):
    output_data = input_data/input_data.max()
    return output_data


# def input_fn(is_training, filename, batch_size=1, num_epochs=1):
#     def example_parser(serialized_example):
#
#         features = tf.parse_single_example(serialized_example,
#                                            features={
#                                                'image': tf.FixedLenFeature([], tf.string),
#                                                'label': tf.FixedLenFeature([], tf.int64),
#                                                })
#         image = tf.decode_raw(features['image'], tf.float32)
#         label = tf.cast(features['label'], tf.int8)
#         return image, label
#
#     dataset = tf.data.TFRecordDataset([filename])
#     if is_training:
#       dataset = dataset.shuffle(buffer_size=10000)
#     # dataset = dataset.repeat(num_epochs)     #
#     # Map example_parser over dataset, and batch results by up to batch_size
#     dataset = dataset.map(example_parser)
#     dataset = dataset.repeat(num_epochs)     #
#     dataset = dataset.batch(batch_size)
#     iterator = dataset.make_one_shot_iterator()   # iterator 迭代器
#     images, labels = iterator.get_next()
#     return images, labels

# 第一个版本，有问题，增加batch_size 输出数据虽然增加，但是效率却没增加（和cpu对比）
def input_fn(is_training, filename, batch_size, num_epochs=1):
    def example_parser(serialized_example):

        features = tf.parse_single_example(serialized_example,
                                           features={
                                               'image': tf.FixedLenFeature([], tf.string),
                                               'label': tf.FixedLenFeature([], tf.int64),
                                               })
        image = tf.decode_raw(features['image'], tf.float32)
        label = tf.cast(features['label'], tf.int8)
        return image, label

    dataset = tf.data.TFRecordDataset([filename])
    if is_training:
      dataset = dataset.shuffle(buffer_size=10000)
    # dataset = dataset.repeat(num_epochs)     #
    # Map example_parser over dataset, and batch results by up to batch_size
    dataset = dataset.map(example_parser).prefetch(batch_size)
    dataset = dataset.repeat(num_epochs)     #
    dataset = dataset.batch(batch_size)
    iterator = dataset.make_initializable_iterator()   # iterator 迭代器
    # iterator = dataset.make_one_shot_iterator()   # iterator 迭代器
    images, labels = iterator.get_next()
    return images, labels
#
# def input_fn(is_training, filename, batch_size=1, num_epochs=1):
#     def example_parser(serialized_example):
#
#         features = tf.parse_single_example(serialized_example,
#                                            features={
#                                                'image': tf.FixedLenFeature([], tf.string),
#                                                'label': tf.FixedLenFeature([], tf.int64),
#                                                })
#         image = tf.decode_raw(features['image'], tf.float32)
#         label = tf.cast(features['label'], tf.int8)
#         return image, label
#
#     dataset = tf.data.TFRecordDataset([filename])
#     if is_training:
#       dataset = dataset.shuffle(buffer_size=batch_size)
#     # dataset = dataset.repeat(num_epochs)     #
#     # Map example_parser over dataset, and batch results by up to batch_size
#     dataset = dataset.map(example_parser).prefetch(batch_size)
#     dataset = dataset.repeat(num_epochs)     #
#     dataset = dataset.batch(batch_size)
#     iterator = dataset.make_one_shot_iterator()   # iterator 迭代器
#     images, labels = iterator.get_next()
#     return images, labels