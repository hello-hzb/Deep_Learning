# -*- coding: utf-8 -*-
# 20180322 15:14

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from Autoencoder import SAEnoplh, SAEnoplhdevice, SAEnoplhV4
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

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
log_dir = os.path.join(os.getcwd(), 'ship_losangeles/')

# 1 读取高光谱图像数据
# 测试用的plane数据
load_plane = os.path.join(os.getcwd(), '../hyper_data/ship_losangeles.mat')
load_plane_label = os.path.join(os.getcwd(), '../hyper_data/ship_losangeles_label.mat')

plane = sio.loadmat(load_plane)
label = sio.loadmat(load_plane_label)

plane_image = plane['image']/np.max(plane['image'])
plane_image = np.reshape(plane_image, (1818680, 224))
trX = plane_image.reshape(2393, 760, 224)
load_label = label['label'].reshape(1818680)/100


class Modelconfig(object):
    n_samples = trX.shape[0] * trX.shape[1]                       # 样本的个数
    input_size = trX.shape[2]                                     # 输入层节点数
    training_epochs = 1000                                         # 训练的轮数
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

    # sae = SAEnoplh(input_data=image,
    #                n_input=config.input_size,
    #                stack_size=config.stack_size,
    #                hidden_size=config.hidden_size,
    #                optimizer=tf.train.AdamOptimizer(learning_rate=0.00005)
    #                )
else:
    print("Using CPU")
    sae = SAEnoplhdevice(input_data=image,
                         n_input=config.input_size,
                         stack_size=config.stack_size,
                         hidden_size=config.hidden_size,
                         optimizer=tf.train.AdamOptimizer(learning_rate=0.00005)
                         )

init = tf.global_variables_initializer()
# cpu_num = int(os.environ.get('CPU_NUM', 1))
# config1 = tf.ConfigProto(device_count={"CPU": cpu_num},
#                         inter_op_parallelism_threads = cpu_num,
#                         intra_op_parallelism_threads = cpu_num, log_device_placement=True)
# with tf.Session(config=config1) as sess:
with tf.Session(config=tf.ConfigProto(allow_soft_placement=True, device_count={"CPU": 2})) as sess:
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
    # print ("total time is ", (time.time()-start_time)/(total_batch*config.training_epochs))
    coord.request_stop()
    coord.join(threads)


    # 利用重建误差进行AUC计算
    hidden_tmp, recon_err_tmp = sess.run((sae.hidden_out, sae.errtmp), feed_dict={image: plane_image})
    recon_err_tmp = np.sum(recon_err_tmp, axis=1)
    scores = np.array(recon_err_tmp)
    fpr, tpr, thresholds = roc_curve(load_label, scores)  # 输出数据维度和阈值有关，阈值的取值和预测值有关，等于预测值
    roc_auc = roc_auc_score(load_label, scores)  # 计算AUC值
    print(roc_auc)
    # Plot of a ROC curve for a specific class 将结果绘图输出
    plt.figure()
    lw = 2
    plt.plot(fpr, tpr, color='darkorange', lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.show()
