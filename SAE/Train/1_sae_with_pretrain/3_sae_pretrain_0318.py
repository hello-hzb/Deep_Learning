# -*- coding: utf-8 -*-
# 20180318
# 旧版ae

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from Autoencoder import SAEnoplh, SAEnoplhdevice, SAEnoplhV2
from data_preprocess import get_random_block_from_data
from read_data import input_fn
from tensorflow.python.client import timeline
from sklearn.metrics import roc_curve, auc, roc_auc_score
import matplotlib.pyplot as plt
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
trX = plane_image.reshape(100, 100, 126)
load_label = label['array']


class Modelconfig(object):
    n_samples = trX.shape[0] * trX.shape[1]                       # 样本的个数
    input_size = trX.shape[2]                                     # 输入层节点数
    training_epochs = 100                                         # 训练的轮数
    batch_size = 100                                               # 更新一次参数训练的样本数
    display_step = 1                                              # 打印信息的步数
    stack_size = 3                                                # 自编码器的隐藏层层数
    hidden_size = [166, 20, 166]                                  # 各个隐藏层的节点数
    n_hidden_size = hidden_size[int(math.floor(stack_size / 2))]  # coding层的输出维度
    pretrain_lr = [0.005, 0.005, 0.005, 0.005, 0.005]             # 预训练不同隐藏层之间的学习率
    finetune_lr = 0.00005                                         # 微调训练的学习率
    is_pretrain = 1                                               # 是否预训练开关
    transfer_function = tf.nn.softplus                            # 激活函数


# 建立sae图
config = Modelconfig()
image = tf.placeholder(tf.float32, [None, config.input_size])

use_gpu = 1
if use_gpu:
    ae_function = SAEnoplh         # gpu
else:
    ae_function = SAEnoplhdevice   # cpu

stack_autoencoder = []
for i in xrange(config.stack_size):
    if i == 0:
        image = tf.placeholder(tf.float32, [None, config.input_size])
        ae = ae_function(input_data=image,
                         n_input=config.input_size,
                         stack_size=config.stack_size,
                         hidden_size=config.hidden_size,
                         optimizer=tf.train.AdamOptimizer(learning_rate=0.00005)
                         )

    else:
        ae = ae_function(input_data=stack_autoencoder[i-1].hidden_out,
                         n_input=config.input_size,
                         stack_size=config.stack_size,
                         hidden_size=config.hidden_size,
                         optimizer=tf.train.AdamOptimizer(learning_rate=0.00005)
                         )
    stack_autoencoder.append(ae)


use_gpu = 1
if use_gpu:
    print ("Using GPU")
    # sae = SAEnoplhV2(input_data=image,
    #                  n_input=config.input_size,
    #                  stack_size=config.stack_size,
    #                  hidden_size=config.hidden_size,
    #                  optimizer=tf.train.AdamOptimizer(learning_rate=0.00005)
    #                  )  # 将权重和偏置放到定义到CPU中，性能有10%的提升

    sae = SAEnoplh(input_data=image,
                   n_input=config.input_size,
                   stack_size=config.stack_size,
                   hidden_size=config.hidden_size,
                   optimizer=tf.train.AdamOptimizer(learning_rate=0.00005)
                   )
else:
    print("Using CPU")
    sae = SAEnoplhdevice(input_data=image,
                         n_input=config.input_size,
                         stack_size=config.stack_size,
                         hidden_size=config.hidden_size,
                         optimizer=tf.train.AdamOptimizer(learning_rate=0.00005)
                         )

init = tf.global_variables_initializer()

with tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=True)) as sess:
    sess.run(init)
    summary_writer = tf.summary.FileWriter(log_dir, sess.graph)  # 创建图写入器并写文件
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)
    print("Finetune starting!")
    start_time = time.time()
    for epoch in range(config.training_epochs):
        avg_cost = 0.
        total_batch = int(config.n_samples / config.batch_size)
        for _ in range(total_batch):
            batch_xs = get_random_block_from_data(plane_image, config.batch_size)
            cost, _ = sess.run((sae.cost, sae.optimizer),
                               feed_dict={image: batch_xs})

            avg_cost += cost / config.n_samples * config.batch_size
        # Display logs per epoch step
        if epoch % config.display_step == 0:
            print("Epoch:", '%d,' % (epoch + 1), "Cost:", "{:.9f}".format(avg_cost),
                  "Time/Epoch is ", (time.time() - start_time))
    print ("total time is ", time.time()-start_time)
    coord.request_stop()
    coord.join(threads)


