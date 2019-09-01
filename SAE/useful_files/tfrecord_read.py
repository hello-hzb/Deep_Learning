# -*- coding: utf-8 -*-
# 20171115
# 读取tfrecord文件的方式
import tensorflow as tf
import os
num_image = 10000
filename = "/home/ubuntu/my_file/hyper_sae/gpu_work/tf_record_data/plane.tf_records"
log_dir = os.path.join(os.getcwd(), 'log_dir_tf_read/')


def input_fn(is_training, filename, batch_size=1, num_epochs=1):
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
    dataset = dataset.shuffle(buffer_size=num_image)

  dataset = dataset.repeat(num_epochs)     #

  # Map example_parser over dataset, and batch results by up to batch_size
  dataset = dataset.map(example_parser).prefetch(batch_size)
  dataset = dataset.batch(batch_size)
  iterator = dataset.make_one_shot_iterator()   # iterator 迭代器
  images, labels = iterator.get_next()
  return images, labels

image, label = input_fn(True, filename, batch_size=100, num_epochs=10)

with tf.Session() as sess:  # 开始一个会话
    summary_writer = tf.summary.FileWriter(log_dir, sess.graph)

    init_op = tf.global_variables_initializer()
    sess.run(init_op)
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)
    for i in xrange(1000):
        example, l = sess.run([image, label])  # 在会话中取出image和label
        print(example, l)

    coord.request_stop()
    coord.join(threads)