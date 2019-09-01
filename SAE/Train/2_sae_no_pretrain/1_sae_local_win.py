# -*- coding: utf-8 -*-
# 20180427
# 好的检测精度能达到0.87
# 增加基于Tensorflow的参数保存以及重新加载的功能, 通过键盘输入0或者1进行模式选择
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
log_dir = os.path.join(os.getcwd(), '1_sae_local_win/')

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
    training_epochs = 200                                         # 训练的轮数
    batch_size = 9999                                               # 更新一次参数训练的样本数
    display_step = 1                                              # 打印信息的步数
    stack_size = 3                                                # 自编码器的隐藏层层数
    hidden_size = [166, 20, 166]                                  # 各个隐藏层的节点数
    n_hidden_size = hidden_size[int(math.floor(stack_size / 2))]  # coding层的输出维度
    pretrain_lr = [0.005, 0.005, 0.005, 0.005, 0.005]             # 预训练不同隐藏层之间的学习率
    finetune_lr = 0.005                                         # 微调训练的学习率
    is_pretrain = 1                                               # 是否预训练开关
    is_retrain = input("please input mode, 1 to retrain, 0 to restore ")
    transfer_function = tf.nn.softplus                            # 激活函数


# 建立sae图
config = Modelconfig()
image = tf.placeholder(tf.float32, [None, config.input_size])
# cpu:41.14       gpu:20.82   gpuv2:15.01  cpuv2:43.30
use_gpu = 1
if use_gpu:
    print ("Using GPU")
    sae = SAEnoplhV4(input_data=image,
                     n_input=config.input_size,
                     stack_size=config.stack_size,
                     hidden_size=config.hidden_size,
                     optimizer=tf.train.AdamOptimizer(learning_rate=0.00005)
                     )  # 将权重和偏置放到定义到CPU中，性能有10%的提升


else:
    print("Using CPU")
    sae = SAEnoplhdevice(input_data=image,
                         n_input=config.input_size,
                         stack_size=config.stack_size,
                         hidden_size=config.hidden_size,
                         optimizer=tf.train.AdamOptimizer(learning_rate=0.00005)
                         )

saver = tf.train.Saver(tf.global_variables())
init = tf.global_variables_initializer()
checkpoint_path = os.path.join(log_dir, 'model.ckpt')

with tf.Session() as sess:
    # 如果不进行重新训练，则进入模型重载模式
    if config.is_retrain == 0:
        saver.restore(sess, checkpoint_path)
        print("Model restored")
    # 否则进行模型重新训练
    else:
        sess.run(init)
        # summary_writer = tf.summary.FileWriter(log_dir, sess.graph)  # 创建图写入器并写文件
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        print("Finetune starting!")
        for epoch in range(config.training_epochs):
            # sae.current_step = epoch
            start_time = time.time()
            avg_cost = 0.
            total_batch = int(config.n_samples / config.batch_size)
            for _ in range(total_batch):
                batch_xs = get_random_block_from_data(plane_image, config.batch_size)
                cost, _, = sess.run((sae.cost, sae.optimizer),
                                    feed_dict={image: batch_xs})

                avg_cost += cost / config.n_samples * config.batch_size
            # Display logs per epoch step
            if epoch % config.display_step == 0:
                print("Epoch:", '%d,' % (epoch + 1), "Cost:", "{:.9f}".format(avg_cost),
                      "Time/Epoch is ", (time.time() - start_time))

        # while True:
        #     key = raw_input("Save model? y/n")
        #     if key == 'y':
        #         saver.save(sess, checkpoint_path)
        #         break
        #     elif key == 'n':
        #         break
        saver.save(sess, checkpoint_path)

        coord.request_stop()
        coord.join(threads)

    # 检测器功能
    hidden_tmp, recon_err_tmp = sess.run((sae.hidden_out, sae.errtmp), feed_dict={image: plane_image})
    hidden_tmp = np.reshape(hidden_tmp, (trX.shape[0], trX.shape[1], hidden_tmp.shape[1]))
    recon_err_tmp = np.reshape(recon_err_tmp, (trX.shape[0], trX.shape[1], recon_err_tmp.shape[1]))
    hidden_pad = np.pad(hidden_tmp, pad_width=((local_win_width,), (local_win_width,), (0,)), mode='symmetric')
    recon_err_pad = np.pad(recon_err_tmp, pad_width=((local_win_width,), (local_win_width,), (0,)), mode='symmetric')

    hidden_pad_win = []
    recon_err_pad_win = []
    for i in range(local_win_width, hidden_pad.shape[0] - local_win_width):
        for j in range(local_win_height, hidden_pad.shape[1] - local_win_height):
            hidden_pad_win.append(OneWindow_Edge_Sel(hidden_pad, local_win_width, local_win_height, i, j))
            recon_err_pad_win.append(OneWindow_Edge_Sel(recon_err_pad, local_win_width, local_win_height, i, j))
    hidden_local = np.array(hidden_pad_win, dtype=np.float32)          # 10000*17*32  10000个特征矩阵的窗
    recon_err_local = np.array(recon_err_pad_win, dtype=np.float32)    # 10000*17*126 10000个重建误差的窗
    recon_err_local = np.sum(recon_err_local, axis=2)                  # 10000*17     计算每一个点的重建误差

    scores = []
    for k in xrange(hidden_local.shape[0]):
        ones = np.ones((hidden_local.shape[1], hidden_local.shape[2]))  # 创建全1矩阵
        last_point = hidden_local[k, hidden_local.shape[1] - 1, :]      # 取出第一个窗内的最后的最后一个点
        filter_win = np.multiply(ones, last_point)                      # 单位矩阵所有列和最后一个点一样
        score_tmp = np.sum(np.square(filter_win - hidden_local[k]), axis=1)/recon_err_local[k] # 计算窗外边缘的点与中心点的欧氏距离
        scores.append(np.sum(score_tmp))

    scores = np.array(scores)
    roc_auc1 = roc_auc_score(load_label, scores)
    print ("AUC is ", roc_auc1)



