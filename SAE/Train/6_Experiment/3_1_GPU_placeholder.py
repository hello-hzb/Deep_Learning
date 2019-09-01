# -*- coding: utf-8 -*-
# 20180425
# 增加基于Tensorflow的参数保存以及重新加载的功能, 通过键盘输入0或者1进行模式选择,不要局部窗的设计
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from Autoencoder import SAEnoplhV2, SAEnoplhV4, SAEnoplhdevice
from data_preprocess import get_random_block_from_data, OneWindow_Edge_Sel
from sklearn.metrics import roc_curve, auc, roc_auc_score
import matplotlib.pyplot as plt
import tensorflow as tf
import scipy.io as sio
import numpy as np
import time
import math
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
log_dir = os.path.join(os.getcwd(), '3_1_GPU_placeholder/')

# 1 读取高光谱图像数据
# 测试用的plane数据
load_plane = os.path.join(os.getcwd(), '../hyper_data/plane.mat')
load_plane_label = os.path.join(os.getcwd(), '../hyper_data/plane_label.mat')

plane = sio.loadmat(load_plane)
label = sio.loadmat(load_plane_label)

plane_image = plane['array']
trX = plane_image.reshape(100, 100, 126)
load_label = label['array'].reshape(10000)/255

local_win_width = 2     # 3*2  window is 7*7
local_win_height = 2    # 3*2  window is 7*7


class Modelconfig(object):
    n_samples = trX.shape[0] * trX.shape[1]                       # 样本的个数
    input_size = trX.shape[2]                                     # 输入层节点数
    training_epochs = 10                                         # 训练的轮数
    batch_size = 100                                               # 更新一次参数训练的样本数
    display_step = 1                                              # 打印信息的步数
    stack_size = 3                                                # 自编码器的隐藏层层数
    hidden_size = [20, 20, 20]                                  # 各个隐藏层的节点数
    n_hidden_size = hidden_size[int(math.floor(stack_size / 2))]  # coding层的输出维度
    pretrain_lr = [0.005, 0.005, 0.005, 0.005, 0.005]             # 预训练不同隐藏层之间的学习率
    finetune_lr = 0.0005                                         # 微调训练的学习率
    is_pretrain = 1                                               # 是否预训练开关
    is_retrain = input("please input mode, 1 to retrain, 0 to restore ")
    transfer_function = tf.nn.softplus                            # 激活函数


# 建立sae图
config = Modelconfig()
start_time = time.time()
for i in xrange(100):
    config.hidden_size[0] = config.hidden_size[0] + 1
    config.hidden_size[1] = config.hidden_size[1] + 1
    config.hidden_size[2] = config.hidden_size[2] + 1
    image = tf.placeholder(tf.float32, [None, config.input_size])
    # cpu:41.14       gpu:20.82   gpuv2:15.01  cpuv2:43.30
    use_gpu = 1
    print("Start model", i)
    with tf.variable_scope("Model%d" % i):
        if use_gpu:
            print ("Using GPU")
            sae = SAEnoplhV4(input_data=image,
                             n_input=config.input_size,
                             stack_size=config.stack_size,
                             hidden_size=config.hidden_size,
                             optimizer=tf.train.AdamOptimizer(learning_rate=0.00005)
                             )  # 将权重和偏置放到定义到CPU中，性能有10%的提升
            print('GPU0: ', config.hidden_size)

    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        print("Finetune starting!")
        for epoch in range(config.training_epochs):
            # start_time = time.time()
            avg_cost = 0.
            total_batch = int(config.n_samples / config.batch_size)
            for _ in range(total_batch):
                batch_xs = get_random_block_from_data(plane_image, config.batch_size)
                cost, _, = sess.run((sae.cost, sae.optimizer),
                                    feed_dict={image: batch_xs})

                avg_cost += cost / config.n_samples * config.batch_size
            # Display logs per epoch step
            # if epoch % config.display_step == 0:
            #     print("Epoch:", '%d,' % (epoch + 1), "Cost:", "{:.9f}".format(avg_cost),
            #           "Time/Epoch is ", (time.time() - start_time))

        coord.request_stop()
        coord.join(threads)

print ("End, total time is ", time.time()-start_time)



