# -*- coding: utf-8 -*-
# 20180526
# 增加基于Tensorflow的参数保存以及重新加载的功能, 通过键盘输入0或者1进行模式选择,不要局部窗的设计
# SAE算法和RX算法的检测结果对比，检查图像为美国西海岸海域图像
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

# import matplotlib as mpl
from matplotlib.font_manager import FontProperties
plt.rcParams['font.sans-serif'] = ['NSimSun'] # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False    # 用来正常显示负号
font = FontProperties(fname="/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc", size=13)


os.environ["CUDA_VISIBLE_DEVICES"] = "1"
log_dir = os.path.join(os.getcwd(), '6_ship_SAE_RXD/')


# 1 读取高光谱图像数据
# 测试用的plane数据
load_plane = os.path.join(os.getcwd(), '../hyper_data/ship_losangeles.mat')
load_plane_label = os.path.join(os.getcwd(), '../hyper_data/ship_losangeles_label.mat')

plane = sio.loadmat(load_plane)
label = sio.loadmat(load_plane_label)

plane_image = plane['image']/np.max(plane['image'])
plt.imshow(plane_image[:, :, 10])
plt.show()
plane_image = np.reshape(plane_image, (1818680, 224))
trX = plane_image.reshape(2393, 760, 224)
load_label = label['label'].reshape(1818680)/100
plt.imshow(label['label'])
plt.show()


class Modelconfig(object):
    n_samples = trX.shape[0] * trX.shape[1]                       # 样本的个数
    input_size = trX.shape[2]                                     # 输入层节点数
    training_epochs = 300                                         # 训练的轮数
    batch_size = 10000                                               # 更新一次参数训练的样本数
    display_step = 1                                              # 打印信息的步数
    stack_size = 3                                                # 自编码器的隐藏层层数
    hidden_size = [166, 20, 166]                                  # 各个隐藏层的节点数
    n_hidden_size = hidden_size[int(math.floor(stack_size / 2))]  # coding层的输出维度
    pretrain_lr = [0.005, 0.005, 0.005, 0.005, 0.005]             # 预训练不同隐藏层之间的学习率
    finetune_lr = 0.0005                                         # 微调训练的学习率
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
        variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
        a = sess.run(variables)
        summary_writer = tf.summary.FileWriter(log_dir, sess.graph)  # 创建图写入器并写文件
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

        saver.save(sess, checkpoint_path)
        coord.request_stop()
        coord.join(threads)

    # 利用重建误差进行AUC计算
    reconstruction, recon_err_tmp = sess.run((sae.reconstruction, sae.errtmp), feed_dict={image: plane_image})
    recon_err_tmp = np.sum(recon_err_tmp, axis=1)
    scores = np.array(recon_err_tmp)
    # 计算SAE检测洛杉矶西海岸图像的AUC值
    fpr, tpr, thresholds = roc_curve(load_label, scores)  # 输出数据维度和阈值有关，阈值的取值和预测值有关，等于预测值
    roc_auc = roc_auc_score(load_label, scores)           # 计算AUC值
    print (roc_auc)

    # 计算RXD检测洛杉矶西海岸图像的AUC值
    ship_rxd_score = sio.loadmat("../7_matlab_work/ship_score.mat")
    rxd_score = ship_rxd_score["ship_score"]
    rxd_fpr, rxd_tpr, rxd_thresholds = roc_curve(load_label, rxd_score)  # 输出数据维度和阈值有关，阈值的取值和预测值有关，等于预测值
    rxd_roc_auc = roc_auc_score(load_label, rxd_score)  # 计算AUC值
    print (" RXD AUC is ", rxd_roc_auc)

    # Plot of a ROC curve for a specific class 将结果绘图输出
    plt.figure("Ship")
    lw = 2
    plt.plot(fpr, tpr, color='darkorange', lw=lw, label=U'GPU-SAE ROC (AUC = %0.4f)' % roc_auc)
    # plt.plot(rxd_fpr, rxd_tpr, color='navy', lw=lw, linestyle=':', label=U'RXD ROC (AUC = %0.4f)' % rxd_roc_auc)

    # plt.plot([0, 1], [0, 1], color='k', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.xlabel(U'假正例率', fontproperties=font)
    plt.ylabel(U'真正例率', fontproperties=font)
    # plt.title('Receiver operating characteristic')
    plt.legend(loc="lower right", fontsize=13)
    plt.show()
