# -*- coding: utf-8 -*-
# 20171123 11:11
# 计算过程分为3个类, 增加自定义隐藏层
# 基本计算过程正确

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from Autoencoder import Autoencoderpure, Anomalydetect, Libcal, SAE
from data_preprocess import OneWindow_Edge_Sel, get_random_block_from_data
from sklearn.metrics import roc_curve, auc, roc_auc_score
import matplotlib.pyplot as plt
import tensorflow as tf
import scipy.io as sio
import numpy as np
import time
import math
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
log_dir = os.path.join(os.getcwd(), 'log_dir_AD/')
##################################################
# 0 paramter initial ###########################
local_win_width = 2     # 3*2  window is 7*7
local_win_height = 2    # 3*2  window is 7*7
##############################################
# 1   读取高光谱图像数据

# 测试用的plane数据
load_plane = os.path.join(os.getcwd(), '../hyper_data/plane.mat')
load_plane_label = os.path.join(os.getcwd(), '../hyper_data/plane_label.mat')
plane = sio.loadmat(load_plane)
label = sio.loadmat(load_plane_label)

plane_image = plane['array']
trX = plane_image.reshape(100, 100, 126)
load_label = label['array']


# #  5 Initial SAE ###############
class Modelconfig(object):
    n_samples = trX.shape[0] * trX.shape[1]             # 样本的个数
    input_size = trX.shape[2]                                # 输入层节点数
    training_epochs = 30                                          # 训练的轮数
    batch_size = 100                                              # 更新一次参数训练的样本数
    display_step = 1                                              # 打印信息的步数
    stack_size = 3                                                # 自编码器的隐藏层层数
    hidden_size = [166, 20, 166]                                  # 各个隐藏层的节点数
    n_hidden_size = hidden_size[int(math.floor(stack_size / 2))]  # coding层的输出维度
    pretrain_lr = [0.00005, 0.00005, 0.00005, 0.00005, 0.00005]   # 预训练不同隐藏层之间的学习率
    finetune_lr = 0.00005                                         # 微调训练的学习率
    is_pretrain = 1                                               # 是否预训练开关
    transfer_function = tf.nn.softplus                            # 激活函数


# ###############################################
# 建立sae图
config = Modelconfig()
with tf.name_scope(name="Detection_System"):
    with tf.name_scope(name="Pretraining_Model"):
        sae = []
        for i in xrange(config.stack_size):
            with tf.variable_scope('Autoencoder_%d' % i):
                if i == 0:
                    autoencoder = Autoencoderpure(
                        n_input=config.input_size,
                        n_hidden=config.hidden_size[i],
                        transfer_function=tf.nn.softplus,
                        optimizer=tf.train.AdamOptimizer(learning_rate=config.pretrain_lr[i]))
                else:
                    autoencoder = Autoencoderpure(
                        n_input=config.hidden_size[i-1],
                        n_hidden=config.hidden_size[i],
                        transfer_function=tf.nn.softplus,
                        optimizer=tf.train.AdamOptimizer(learning_rate=config.pretrain_lr[i]))
                sae.append(autoencoder)

    with tf.name_scope(name="Fintune"):
        stack_autoencoder = SAE(sae_tuple=sae,
                                stack_size=config.stack_size,
                                hidden_size=config.hidden_size,
                                n_input=config.input_size,
                                transfer_function=tf.nn.softplus,
                                optimizer=tf.train.AdamOptimizer(learning_rate=config.finetune_lr))

# ### Train with one sample #########
init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    hidden_future = []
    # 单次数据训练
    if config.is_pretrain:
        x_train = []
        print("Pretrain starting!")
        for j in xrange(config.stack_size):
            if j == 0:
                x_train = plane_image
            else:
                x_train = sae[j-1].transform(X=x_train, sess=sess)
                print (x_train.shape)
            for epoch in xrange(config.training_epochs):
                start_time = time.time()
                avg_cost = 0.
                total_batch = int(config.n_samples / config.batch_size)
                for _ in range(total_batch):
                    batch_xs = get_random_block_from_data(x_train, config.batch_size)
                    cost = sae[j].partial_fit(X=x_train, sess=sess)  # 开始训练
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
                cost = stack_autoencoder.partial_fit(X=batch_xs, sess=sess)
                avg_cost += cost / config.n_samples * config.batch_size

                # Display logs per epoch step
            if epoch % config.display_step == 0:
                print("Epoch:", '%d,' % (epoch + 1), "Cost:", "{:.9f}".format(avg_cost),
                      "Time/Epoch is ", (time.time() - start_time))
#
# with tf.Session() as sess:
#     sess.run(init)
#     hidden_future = []
#     x_train = []
#
#     print("Pretrain starting!")
#     for j in xrange(config.stack_size):
#         if j == 0:
#             x_train = trX
#         else:
#             x_train_pre = x_train
#             x_train = sae[j-1].transform(session=sess, X=x_train_pre)
#             print (x_train.shape)
#             hidden_future.append(x_train)
#
#         for epoch in xrange(training_epochs):
#             avg_cost = 0.
#             total_batch = int(n_samples / batch_size)
#             for _ in range(total_batch):
#                 batch_xs = get_random_block_from_data(x_train, batch_size)
#                 # Fit training using batch data
#                 cost = sae[j].partial_fit(session=sess, X=batch_xs)  # 开始训练
#                 # Compute average loss每次计算一个批次误差,所有批次误差加权平均后是所有样本的平均误差
#                 avg_cost += cost / n_samples * batch_size
#
#             # Display logs per epoch step
#             if epoch % display_step == 0:
#                 print("Epoch:", '%d,' % (epoch + 1),
#                       "Cost:", "{:.9f}".format(avg_cost))
#
#     print ("Finetune starting!")
#     for epoch in range(1000):
#         avg_cost = 0.
#         total_batch = int(n_samples / batch_size)
#         for _ in range(total_batch):
#             batch_xs = get_random_block_from_data(trX, batch_size)
#             cost = stack_autoencoder.partial_fit(session=sess, X=batch_xs)
#             avg_cost += cost / n_samples * batch_size
#
#         # Display logs per epoch step
#         if epoch % display_step == 0:
#             print("Epoch:", '%d,' % (epoch + 1),
#                   "Cost:", "{:.9f}".format(avg_cost))
#
#
#
#
