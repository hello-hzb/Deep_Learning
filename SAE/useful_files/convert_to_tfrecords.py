# -*- coding: utf-8 -*-
# 20171110
# 将mat文件的数据成tfrecords文件
# 用户自定义参数：image_dir, label_dir, record_dir, tf_record_name
#
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tensorflow as tf
import scipy.io as sio

image_dir = '../hyper_data3/plane.mat'                      # image 的路径+文件名***************用户自定义
label_dir = '../hyper_data3/plane_label.mat'                # label 的路径+文件名***************用户自定义
record_dir = os.path.join(os.getcwd(), 'tf_record_data/')   # 指定保存tfrecords文件的路径********用户自定义
tf_record_name = "plane.tf_records"                         # 指定要生成的文件名*****************用户自定义
if not os.path.exists(record_dir):                          # 判断保存tfrecords文件的路径的路径是否存在
    os.mkdir(record_dir)                                    # 如果不存在保存路径则创建
output_file = os.path.join(record_dir, tf_record_name)      # 生成的路径+文件名


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[str(value)]))


def convert_hyper_to_tfrecord(image, label, output_file):
    """Converts a file to TFRecords."""
    print('Generating %s' % output_file)
    with tf.python_io.TFRecordWriter(output_file) as record_writer:
        num_entries_in_batch = len(label)
        for i in range(num_entries_in_batch):
            example = tf.train.Example(features=tf.train.Features(
                feature={
                    'image': _bytes_feature(image[i].tobytes()),
                    'label': _int64_feature(label[i])
                    }))
            record_writer.write(example.SerializeToString())


def read_image(dir_filename):
    load_data = sio.loadmat(dir_filename)
    load_matrix = load_data['array']   # 根据不同的数据，需要改变一下tuple的名字
    return load_matrix


def read_tfrecord(tfrecords_file):
    pass


def main():
    image = read_image(image_dir)
    label_temp = read_image(label_dir)
    label = label_temp.reshape(label_temp.shape[0]*label_temp.shape[1], -1)
    try:
        os.remove(output_file)
    except OSError:
        pass
    # Convert to tf.train.Example and write the to TFRecords.
    convert_hyper_to_tfrecord(image, label, output_file)
    print ("It done")



