# -*- coding: utf-8 -*-
# 20180322
# 实现SAE并做预训练
# 之前的版本中输出层没有单独预训练，本次加入输出层的预训练

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from Autoencoder import AE_mult_add, Stack_autoencoder
from data_preprocess import get_random_block_from_data
import tensorflow as tf
import scipy.io as sio
import numpy as np
import time
import math
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
log_dir = os.path.join(os.getcwd(), 'log_dir_gpu/')

# 1 读取高光谱图像数据
# 测试用的plane数据
load_plane = os.path.join(os.getcwd(), '../hyper_data/plane.mat')
load_plane_label = os.path.join(os.getcwd(), '../hyper_data/plane_label.mat')

plane = sio.loadmat(load_plane)
label = sio.loadmat(load_plane_label)

plane_image = plane['array']
# trX = plane_image.reshape(100, 100, 126)
load_label = label['array']


class Modelconfig(object):
    n_samples = plane_image.shape[0]                       # 样本的个数
    input_size = plane_image.shape[1]                                     # 输入层节点数
    training_epochs = 100                                         # 训练的轮数
    batch_size = 100                                               # 更新一次参数训练的样本数
    display_step = 1                                              # 打印信息的步数
    stack_size = 3                                                # 自编码器的隐藏层层数
    hidden_size = [166, 20, 166, 20, 166]                                  # 各个隐藏层的节点数
    n_hidden_size = hidden_size[int(math.floor(stack_size / 2))]  # coding层的输出维度
    pretrain_lr = [0.005, 0.005, 0.005, 0.005, 0.005]             # 预训练不同隐藏层之间的学习率
    finetune_lr = 0.005                                         # 微调训练的学习率
    is_pretrain = 1                                               # 是否预训练开关
    transfer_function = tf.nn.softplus                            # 激活函数


# 建立sae图
config = Modelconfig()
# image = tf.placeholder(tf.float32, [None, config.input_size])
stack_autoencoder = []
for i in xrange(config.stack_size):
    with tf.variable_scope('pretrain_model%d' % i):
        if i == 0:
            ae = AE_mult_add(n_input=config.input_size,
                             hidden_size=config.hidden_size[0],
                             optimizer=tf.train.AdamOptimizer(config.finetune_lr))
            stack_autoencoder.append(ae)

        else:

            ae = AE_mult_add(n_input=config.hidden_size[i-1],
                             hidden_size=config.hidden_size[i],
                             optimizer=tf.train.AdamOptimizer(config.finetune_lr))
            stack_autoencoder.append(ae)

with tf.variable_scope('output_layer'):
    ae = AE_mult_add(n_input=config.hidden_size[config.stack_size-1],
                     hidden_size=config.input_size,
                     optimizer=tf.train.AdamOptimizer(config.finetune_lr))
    stack_autoencoder.append(ae)

sae = Stack_autoencoder(sae_tuple=stack_autoencoder,
                        stack_size=config.stack_size,
                        n_input=config.input_size,
                        optimizer=tf.train.AdamOptimizer(config.finetune_lr))

init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    # hidden_future = []
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)
    # 单次数据训练
    if config.is_pretrain:
        x_train = []
        print("Pretrain starting!")
        for j in xrange(config.stack_size+1):
            if j == 0:
                x_train = plane_image
            else:
                x_train = sess.run(stack_autoencoder[j-1].hidden,
                                   feed_dict={stack_autoencoder[j-1].input_data: x_train})
                print (x_train.shape)

            for epoch in xrange(config.training_epochs):
                start_time = time.time()
                avg_cost = 0.
                total_batch = int(config.n_samples / config.batch_size)
                for _ in range(total_batch):
                    batch_xs = get_random_block_from_data(x_train, config.batch_size)
                    cost, _ = sess.run((stack_autoencoder[j].cost, stack_autoencoder[j].optimizer),
                                       feed_dict={stack_autoencoder[j].input_data: batch_xs})
                    avg_cost += cost / config.n_samples * config.batch_size

                    # Display logs per epoch step
                if epoch % config.display_step == 0:
                    print("Epoch:", '%d,' % (epoch + 1), "Cost:", "{:.9f}".format(avg_cost),
                          "Time/Epoch is ", (time.time() - start_time))

    print("Finetune starting!")
    for epoch in xrange(config.training_epochs):
        start_time = time.time()
        avg_cost = 0.
        total_batch = int(config.n_samples / config.batch_size)
        for _ in range(total_batch):
            batch_xs = get_random_block_from_data(plane_image, config.batch_size)
            cost, _ = sess.run((sae.cost, sae.optimizer),
                               feed_dict={sae.input_data: batch_xs})
            avg_cost += cost / config.n_samples * config.batch_size

            # Display logs per epoch step
        if epoch % config.display_step == 0:
            print("Epoch:", '%d,' % (epoch + 1), "Cost:", "{:.9f}".format(avg_cost),
                  "Time/Epoch is ", (time.time() - start_time))

    coord.request_stop()
    coord.join(threads)




#
# with tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=True)) as sess:
#     sess.run(init)
#     summary_writer = tf.summary.FileWriter(log_dir, sess.graph)  # 创建图写入器并写文件
#
#     coord = tf.train.Coordinator()
#     threads = tf.train.start_queue_runners(coord=coord)
#     print("Finetune starting!")
#     start_time = time.time()
#     for epoch in range(config.training_epochs):
#         avg_cost = 0.
#         total_batch = int(config.n_samples / config.batch_size)
#         for _ in range(total_batch):
#             batch_xs = get_random_block_from_data(plane_image, config.batch_size)
#             loss, _ = sess.run((sae.cost, sae.optimizer), feed_dict={sae.input_data: batch_xs})
#
#             avg_cost += loss / config.n_samples * config.batch_size
#         # Display logs per epoch step
#         if epoch % config.display_step == 0:
#             print("Epoch:", '%d,' % (epoch + 1), "Cost:", "{:.9f}".format(avg_cost),
#                   "Time/Epoch is ", (time.time() - start_time))
#     print ("total time is ", time.time()-start_time)
#     coord.request_stop()
#     coord.join(threads)
#
#
