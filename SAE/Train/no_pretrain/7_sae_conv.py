# -*- coding: utf-8 -*-
# 使用卷积操作对整张图片进行卷积，多层网络，修改写法，方便网络层数及卷积核个数调整
# set kernel_size to 1,it is a SAE, but it runs slower than sae,
# batch_size is 100, conv takes 0.26s/epoch and sae takes 0.17s/epoch

# 20180425
# 仅使用重建误差进行AUC计算,增加模型保存和重载, 当前模型20000轮训练能够达到0.8668的AUC精度
# 40000轮训练达到0.875AUC
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from Autoencoder import SAEnoplh, SAEnoplhdevice
from data_preprocess import get_random_block_from_data, OneWindow_Edge_Sel
from read_data import input_fn, write_mat
from tensorflow.python.client import timeline
from sklearn.metrics import roc_curve, auc, roc_auc_score
import matplotlib.pyplot as plt
import tensorflow as tf
import scipy.io as sio
import numpy as np
import time
import math
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
log_dir = os.path.join(os.getcwd(), '7_sae_conv/')

# 1 读取高光谱图像数据
# 测试用的plane数据
load_plane = os.path.join(os.getcwd(), '../hyper_data/plane.mat')
load_plane_label = os.path.join(os.getcwd(), '../hyper_data/plane_label.mat')
data_tfrecord = '/home/ubuntu-server/my_file/master_project/OnlineDetectorOptimize/hyper_data/plane.tf_records'
flag = os.path.join(os.getcwd(), 'performace_log/flag.mat')
flag2 = os.path.join(os.getcwd(), 'performace_log/flag2.mat')  # 用来记录训练到哪一步

plane = sio.loadmat(load_plane)
label = sio.loadmat(load_plane_label)

plane_image = plane['array']
trX = plane_image.reshape(100, 100, 126)
load_label = label['array'].reshape(10000)/255

kk_size = 1
local_win_width = 2     # 3*2  window is 7*7
local_win_height = 2    # 3*2  window is 7*7


class Modelconfig(object):
    n_samples = trX.shape[0] * trX.shape[1]                       # 样本的个数
    input_size = trX.shape[2]                                     # 输入层节点数
    training_epochs = 20000                                 # 训练的轮数
    batch_size = 10000                                              # 更新一次参数训练的样本数
    display_step = 1                                              # 打印信息的步数
    stack_size = 3                                                # 自编码器的隐藏层层数
    hidden_size = [166, 20, 166, 83, 166]                                  # 各个隐藏层的节点数
    kernel_size = [kk_size, kk_size, kk_size, kk_size, kk_size, kk_size, kk_size]
    output_kernel_size = kk_size
    n_hidden_size = hidden_size[int(math.floor(stack_size / 2))]  # coding层的输出维度
    pretrain_lr = [0.005, 0.005, 0.005, 0.005, 0.005]             # 预训练不同隐藏层之间的学习率
    finetune_lr = 0.005                                         # 微调训练的学习率
    is_pretrain = 1                                               # 是否预训练开关
    is_retrain = input("please input mode, 1 to retrain, 0 to restore ")
    transfer_function = tf.nn.softplus                            # 激活函数
    lr = 0.00005


# 建立sae图
config = Modelconfig()
image = tf.placeholder(tf.float32, [trX.shape[0], trX.shape[1], trX.shape[2]])
use_gpu = 1


class CNNetwork(object):
    def __init__(self, inputs, stack_size, kernel_size, hidden_size, using_gpu):
        """Takes the MNIST inputs and mode and outputs a tensor of logits."""
        # Input Layer
        # Reshape X to 4-D tensor: [batch_size, width, height, channels]
        # MNIST images are 28x28 pixels, and have one color channel
        inputs = tf.reshape(inputs, [-1, inputs.shape[0], inputs.shape[1], inputs.shape[2]])

        # When running on GPU, transpose the data from channels_last (NHWC) to
        # channels_first (NCHW) to improve performance.
        # See https://www.tensorflow.org/performance/performance_guide#data_formats
        if using_gpu:
            device_name = "gpu:0"
            data_format = 'channels_first'
            inputs = tf.transpose(inputs, [0, 3, 1, 2])
        else:
            device_name = "cpu:0"
            data_format = 'channels_last'

        with tf.device(device_name):
            hidden = inputs
            for i in xrange(stack_size):
                hidden = tf.layers.conv2d(
                    inputs=hidden,
                    filters=hidden_size[i],
                    kernel_size=[kernel_size[i], kernel_size[i]],
                    padding='same',
                    activation=tf.nn.softplus,
                    data_format=data_format)

                if math.floor(stack_size / 2) == i:
                    # 特征层处理：1.获得特征并转置；2.转置后padding
                    self.hidden_out = hidden
                    hidden_tmp = tf.transpose(self.hidden_out, [0, 2, 3, 1])   # 将中间隐藏层结果转置，让通道数在最后一维
                    self.shape = tf.shape(hidden_tmp)  # 获得转置后的形状
                    self.hidden_pad = tf.pad(tf.reshape(hidden_tmp, [self.shape[1], self.shape[2], self.shape[3]]),
                                             [[local_win_width, local_win_width],
                                              [local_win_height, local_win_height],
                                              [0, 0]],
                                             "SYMMETRIC")  # 对整张图片进行边缘对称pad

            self.conv_out = tf.layers.conv2d(
                inputs=hidden,
                filters=config.input_size,
                kernel_size=[config.output_kernel_size, config.output_kernel_size],
                padding='same',
                activation=tf.nn.softplus,
                data_format=data_format)

            self.cost = 0.5 * tf.reduce_mean(tf.pow(tf.subtract(self.conv_out, inputs), 2.0))
            # 重建误差处理：1.重建误差计算；2.重建误差转置；3重建误差reshape；4.重建误差padding
            self.recon_err_tmp = tf.square(self.conv_out - inputs)
            recon_err_tmp = tf.transpose(self.recon_err_tmp, [0, 2, 3, 1])  # 将中间隐藏层结果转置，让通道数在最后一维
            self.errshape = tf.shape(recon_err_tmp)  # 获得转置后的形状
            self.recon_err = tf.reshape(recon_err_tmp, [self.errshape[1], self.errshape[2], self.errshape[3]])
            self.recon_err_pad = tf.pad(self.recon_err,
                                        [[local_win_width, local_win_width],
                                         [local_win_height, local_win_height],
                                         [0, 0]],
                                        "SYMMETRIC")  # 对整张图片进行边缘对称pad

            optimizer = tf.train.AdamOptimizer(learning_rate=config.lr)
            self.optimizer = optimizer.minimize(self.cost)


cnn_model = CNNetwork(inputs=image,
                      stack_size=config.stack_size,
                      kernel_size=config.kernel_size,
                      hidden_size=config.hidden_size,
                      using_gpu=use_gpu)

init = tf.global_variables_initializer()
saver = tf.train.Saver(tf.global_variables())
checkpoint_path = os.path.join(log_dir, 'model.ckpt')

with tf.Session() as sess:
    # 如果不进行重新训练，则进入模型重载模式
    if config.is_retrain == 0:
        saver.restore(sess, checkpoint_path)
        print("Model restored")
    # 否则进行模型重新训练
    else:
        sess.run(init)
        variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
        a = sess.run(variables)
        # summary_writer = tf.summary.FileWriter(log_dir, sess.graph)  # 创建图写入器并写文件
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        print("Finetune starting!")

        for epoch in range(config.training_epochs):
            start_time = time.time()
            avg_cost = 0.
            total_batch = int(config.n_samples / config.batch_size)
            for _ in range(total_batch):
                cost, _ = sess.run((cnn_model.cost, cnn_model.optimizer), feed_dict={image: trX})
                avg_cost += cost / config.n_samples * config.batch_size
            # Display logs per epoch step
            if epoch % config.display_step == 0:
                print("Epoch:", '%d,' % (epoch + 1), "Cost:", "{:.9f}".format(avg_cost),
                      "Time/Epoch is ", (time.time() - start_time))

        recon_err = sess.run(cnn_model.recon_err, feed_dict={image: trX})  # 需要reshape后列向求和
        recon_err_global = np.sum(recon_err, axis=2)

        roc_auc1 = roc_auc_score(load_label, recon_err_global.reshape(10000))
        print("AUC is ", roc_auc1)
        # confirm_key = raw_input("Do you want to save model? y/n: ")
        while True:
            confirm_key = raw_input("Do you want to save model? y/n: ")
            if confirm_key == 'y':
                print("Starting save model...")
                # saver.save(sess, checkpoint_path)
                print("Save successfully!")
                break
            elif confirm_key == 'n':
                break
        coord.request_stop()
        coord.join(threads)

    recon_err = sess.run(cnn_model.recon_err, feed_dict={image: trX})   # 需要reshape后列向求和
    recon_err_global = np.sum(recon_err, axis=2)

    roc_auc1 = roc_auc_score(load_label, recon_err_global.reshape(10000))
    print("AUC is ", roc_auc1)
    plt.imshow(recon_err_global)
    plt.show()
















    #
    ######################################
    # hidden_pad, recon_err_pad = sess.run((cnn_model.hidden_pad, cnn_model.recon_err_pad), feed_dict={image: trX}) # 获取检测器需要的隐藏层特征和重建误差
    #
    # hidden_pad_win = []
    # recon_err_pad_win = []
    # for i in range(local_win_width, hidden_pad.shape[0] - local_win_width):
    #     for j in range(local_win_height, hidden_pad.shape[1] - local_win_height):
    #         hidden_pad_win.append(OneWindow_Edge_Sel(hidden_pad, local_win_width, local_win_height, i, j))
    #         recon_err_pad_win.append(OneWindow_Edge_Sel(recon_err_pad, local_win_width, local_win_height, i, j))
    # hidden_local = np.array(hidden_pad_win, dtype=np.float32)          # 10000*17*32  10000个特征矩阵的窗
    # recon_err_local = np.array(recon_err_pad_win, dtype=np.float32)    # 10000*17*126 10000个重建误差的窗
    # recon_err_local = np.sum(recon_err_local, axis=2)                  # 10000*17     计算每一个点的重建误差
    #
    # scores = []
    # for k in xrange(hidden_local.shape[0]):
    #     ones = np.ones((hidden_local.shape[1], hidden_local.shape[2]))  # 创建全1矩阵
    #     last_point = hidden_local[k, hidden_local.shape[1] - 1, :]      # 取出第一个窗内的最后的最后一个点
    #     filter_win = np.multiply(ones, last_point)                      # 单位矩阵所有列和最后一个点一样
    #     score_tmp = np.sum(np.square(filter_win - hidden_local[k]), axis=1)/recon_err_local[k] # 计算窗外边缘的点与中心点的欧氏距离
    #     scores.append(np.sum(score_tmp))
    #
    # # plt.imshow(dis_err.reshape(100, 100))
    # # plt.show()
    # scores = np.array(scores)
    # roc_auc1 = roc_auc_score(load_label, scores)
    # print ("AUC is ", roc_auc1)
    # coord.request_stop()
    # coord.join(threads)
    #

