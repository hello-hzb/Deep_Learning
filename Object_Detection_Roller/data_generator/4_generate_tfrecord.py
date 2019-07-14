# coding=utf-8
"""
Usage:
  # From tensorflow/models/
  # Create train data:
  python/python3 generate_tfrecord.py --csv_input=your path to read train.csv  
                                      --images_input=your path to read images
                                      --output_path=your path to write train.record
                                      --label_map_path=your path to read xxx_label_map.pbtxt

  # Create validation data:
  python/python3 generate_tfrecord.py --csv_input=you path to read val.csv  
                                      --images_input=you path to read images
                                      --output_path=you path to write val.record
                                      --label_map_path=your path to read xxx_label_map.pbtxt
"""
from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import os
import io
import pandas as pd
import tensorflow as tf

from PIL import Image
from object_detection.utils import dataset_util
from object_detection.utils import label_map_util
from collections import namedtuple

# flags是用来接受terminal执行的时候带的参数，参数含义如下
# flags.DEFINE_string(参数名, 默认参数值, 参数说明)
# 本来是在命令行中输入参数值的，这里为了能够在pycharm中执行，所以添加了默认值
flags = tf.app.flags
flags.DEFINE_string('csv_input', '/home/ubuntu/SSD/data_generator/roller_val_labels.csv', 'Path to the CSV input')
flags.DEFINE_string('images_input', '/home/ubuntu/SSD/data_generator/JPEGImages/dataset1_val_img/', 'Path to the images input')
flags.DEFINE_string('output_path', '/home/ubuntu/SSD/data_generator/roller_val.record', 'Path to output TFRecord')
flags.DEFINE_string('label_map_path', '/home/ubuntu/SSD/data_generator/roller_map.pbtxt', 'Path to label map proto')
FLAGS = flags.FLAGS


def split(df, group):
    data = namedtuple('data', ['filename', 'object'])
    gb = df.groupby(group)
    return [data(filename, gb.get_group(x)) for filename, x in
            zip(gb.groups.keys(), gb.groups)]


def create_tf_example(group, label_map_dict, images_path):
    with tf.gfile.GFile(os.path.join(
        images_path, '{}'.format(group.filename)), 'rb') as fid:
        encoded_jpg = fid.read()
    encoded_jpg_io = io.BytesIO(encoded_jpg)
    image = Image.open(encoded_jpg_io)
    width, height = image.size
    filename = group.filename.encode('utf8')

    image_format = b'jpg'
    xmins = []
    xmaxs = []
    ymins = []
    ymaxs = []
    classes_text = []
    classes = []
    for index, row in group.object.iterrows():
        xmins.append(row['xmin'] / width)
        xmaxs.append(row['xmax'] / width)
        ymins.append(row['ymin'] / height)
        ymaxs.append(row['ymax'] / height)
        classes_text.append(row['class'].encode('utf8'))
        classes.append(label_map_dict[row['class']])

    tf_example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': dataset_util.int64_feature(height),
        'image/width': dataset_util.int64_feature(width),
        'image/filename': dataset_util.bytes_feature(filename),
        'image/source_id': dataset_util.bytes_feature(filename),
        'image/encoded': dataset_util.bytes_feature(encoded_jpg),
        'image/format': dataset_util.bytes_feature(image_format),
        'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
        'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
        'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
        'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
        'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
        'image/object/class/label': dataset_util.int64_list_feature(classes),
    }))
    return tf_example


def main(_):
    writer = tf.python_io.TFRecordWriter(FLAGS.output_path)
    label_map_dict = label_map_util.get_label_map_dict(FLAGS.label_map_path)
    images_path = FLAGS.images_input
    examples = pd.read_csv(FLAGS.csv_input)
    grouped = split(examples, 'filename')
    for group in grouped:
        tf_example = create_tf_example(group, label_map_dict, images_path)
        writer.write(tf_example.SerializeToString())

    writer.close()
    output_path = FLAGS.output_path
    print('Successfully created the TFRecords: {}'.format(output_path))


if __name__ == '__main__':
    tf.app.run()

