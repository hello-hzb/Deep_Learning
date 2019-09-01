# -*- coding: utf-8 -*-
# 20171123 11:11
# 计算过程分为3个类, 增加自定义隐藏层
# 基本计算过程正确

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from Autoencoder import Autoencoderpure, Anomalydetect, Libcal, SAE
from data_preprocess import OneWindow_Edge_Sel
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

################################################
# 2  归一化######################

#################################
# 3 Padding ####################
trXPadtmp = tf.pad(trX, [[local_win_width, local_win_width], [local_win_height, local_win_height], [0, 0]], "SYMMETRIC")
with tf.Session() as sess:
    trXpad = sess.run(trXPadtmp)

#################################
# 4 reshape to local window#####
# 这一步操作时间特别长
SelWinPixels = []
for i in range(local_win_width, trXpad.shape[0]-local_win_width):
    for j in range(local_win_height, trXpad.shape[1]-local_win_height):
        SelWinPixels.append(OneWindow_Edge_Sel(trXpad, local_win_width, local_win_height, i, j))
trXLocal = np.array(SelWinPixels, dtype=np.float32)
##################################


# #  5 Initial SAE ###############
class Modelconfig(object):
    n_samples = trXLocal.shape[0] * trXLocal.shape[1]             # 样本的个数
    input_size = trXLocal.shape[2]                                # 输入层节点数
    training_epochs = 30                                          # 训练的轮数
    batch_size = trXLocal.shape[1]                                # 更新一次参数训练的样本数
    display_step = 1                                              # 打印信息的步数
    stack_size = 3                                                # 自编码器的隐藏层层数
    hidden_size = [166, 20, 166]                                  # 各个隐藏层的节点数
    n_hidden_size = hidden_size[int(math.floor(stack_size / 2))]  # coding层的输出维度
    pretrain_lr = [0.00005, 0.00005, 0.00005, 0.00005, 0.00005]   # 预训练不同隐藏层之间的学习率
    finetune_lr = 0.00005                                         # 微调训练的学习率
    is_pretrain = 0                                               # 是否预训练开关
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

    with tf.name_scope(name="Anomaly_Detect"):
        anomalydetect = Anomalydetect(batch_size=config.batch_size,
                                      hidden=stack_autoencoder.hidden_out,
                                      err_temp=stack_autoencoder.errtmp)

    with tf.name_scope(name="Lib_Cal"):
        libcal = Libcal(n_hidden=config.n_hidden_size, hidden=stack_autoencoder.hidden_out)

# ### Train with one sample #########
pre_tarin_data = trX.reshape(trX.shape[0]*trX.shape[1], config.input_size)
init = tf.global_variables_initializer()

dis_err = []                                                                          # 初始化distance存储矩阵
dis_lib = {"libcode": [np.zeros(shape=(config.batch_size, config.n_hidden_size))], "dis": [0.0]}    # 初始化目标库
with tf.Session() as sess:
    sess.run(init)
    summary_writer = tf.summary.FileWriter(log_dir, sess.graph)  # 创建图写入器并写文件
    hidden_future = []
    # 单次数据训练
    if config.is_pretrain:
        x_train = []
        print("Pretrain starting!")
        for j in xrange(config.stack_size):
            if j == 0:
                x_train = pre_tarin_data
            else:
                x_train = sae[j-1].transform(X=x_train, sess=sess)
                print (x_train.shape)
            cost = sae[j].partial_fit(X=x_train, sess=sess)  # 开始训练
        print("Finetune starting!")
        cost = stack_autoencoder.partial_fit(X=pre_tarin_data, sess=sess)

    for i in range(trXLocal.shape[0]):
        tmptrXLocal = trXLocal[i, :, :].reshape(trXLocal.shape[1], config.input_size)
        dis_err.append(sess.run(anomalydetect.dis_err, feed_dict={stack_autoencoder.x: tmptrXLocal}))  # 计算AD获得的重建误差
        dis_temp, diflib, dislibtmp, hidden = sess.run((libcal.DisLib_t, libcal.difLib, libcal.disLibtmp,
                                                        stack_autoencoder.hidden_out),
                                                       feed_dict={libcal.dis_lib: dis_lib["libcode"],
                                                                  stack_autoencoder.x: tmptrXLocal})  # 计算库更新过程获得距离和coding的输出

        # print (dis_lib["dis"][-1])
        if dis_temp > dis_lib["dis"][-1]:
            dis_lib["dis"].append(dis_temp)
            dis_lib["libcode"].append(hidden)
            aa = dis_lib["libcode"]
            cost = stack_autoencoder.partial_fit(X=tmptrXLocal, sess=sess)    # 再次训练
        print ("库计算的距离", dis_temp, "库的最后一个值", dis_lib["dis"][-1])

dis_err = np.array(dis_err)
load_label = load_label.reshape(10000)/255
roc_auc1 = roc_auc_score(load_label, dis_err)
print (roc_auc1)
plt.imshow(dis_err.reshape(100, 100))
plt.show()



