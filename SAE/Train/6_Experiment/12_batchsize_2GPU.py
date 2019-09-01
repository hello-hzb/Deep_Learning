# -*- coding: utf-8 -*-
# 20180628
# 双GPU进行超参数寻优设计，变化batch_size并测试计算耗时
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

os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1"
log_dir = os.path.join(os.getcwd(), '12_2GPU_placeolder/')

# 1 读取高光谱图像数据
# 测试用的plane数据
load_plane = os.path.join(os.getcwd(), '../hyper_data/ship_losangeles.mat')
load_plane_label = os.path.join(os.getcwd(), '../hyper_data/ship_losangeles_label.mat')

plane = sio.loadmat(load_plane)
label = sio.loadmat(load_plane_label)

plane_image = plane['image']/np.max(plane['image'])

plane_image = np.reshape(plane_image, (1818680, 224))
plane_image = plane_image[0:131072, :]
trX = plane_image.reshape(512, 256, 224)
load_label = label['label'].reshape(1818680)/100


class Modelconfig(object):
    n_samples = trX.shape[0] * trX.shape[1]                       # 样本的个数
    input_size = trX.shape[2]                                     # 输入层节点数
    training_epochs = 20000                                         # 训练的轮数
    batch_size = 1                                               # 更新一次参数训练的样本数
    display_step = 1                                              # 打印信息的步数
    stack_size = 3                                                # 自编码器的隐藏层层数
    hidden_size = [26, 26, 26]                                  # 各个隐藏层的节点数
    n_hidden_size = hidden_size[int(math.floor(stack_size / 2))]  # coding层的输出维度
    pretrain_lr = [0.005, 0.005, 0.005, 0.005, 0.005]             # 预训练不同隐藏层之间的学习率
    finetune_lr = 0.0005                                         # 微调训练的学习率
    is_pretrain = 1                                               # 是否预训练开关
    is_retrain = input("please input mode, 1 to retrain, 0 to restore ")
    transfer_function = tf.nn.softplus                            # 激活函数


# 建立sae图
config = Modelconfig()
for i in xrange(0, 100, 2):
    image = tf.placeholder(tf.float32, [None, config.input_size])
    # cpu:41.14       gpu:20.82   gpuv2:15.01  cpuv2:43.30
    use_gpu = 1
    print("Start model", i)
    with tf.variable_scope("Model%d" % i):
        if use_gpu:
            print ("Using GPU")
            config.hidden_size[0] = config.hidden_size[0] + 1
            config.hidden_size[1] = config.hidden_size[1] + 1
            config.hidden_size[2] = config.hidden_size[2] + 1
            with tf.variable_scope('GPU0'):
                with tf.device('/gpu:0'):
                    sae0 = SAEnoplhV4(input_data=image,
                                      n_input=config.input_size,
                                      stack_size=config.stack_size,
                                      hidden_size=config.hidden_size,
                                      optimizer=tf.train.AdamOptimizer(learning_rate=0.00005)
                                      )  # 将权重和偏置放到定义到CPU中，性能有10%的提升
                    # print ('GPU0: ', config.hidden_size)

            config.hidden_size[0] = config.hidden_size[0] + 1
            config.hidden_size[1] = config.hidden_size[1] + 1
            config.hidden_size[2] = config.hidden_size[2] + 1
            with tf.variable_scope('GPU1'):
                with tf.device('/gpu:1'):
                    sae1 = SAEnoplhV4(input_data=image,
                                      n_input=config.input_size,
                                      stack_size=config.stack_size,
                                      hidden_size=config.hidden_size,
                                      optimizer=tf.train.AdamOptimizer(learning_rate=0.00005)
                                      )  # 将权重和偏置放到定义到CPU中，性能有10%的提升
                    # print ('GPU1: ', config.hidden_size)

        init = tf.global_variables_initializer()

        with tf.Session() as sess:
            # 如果不进行重新训练，则进入模型重载模式
            config.batch_size = 1
            for _ in xrange(16):
                start_time = time.time()
                config.batch_size = config.batch_size * 2
                sess.run(init)
                coord = tf.train.Coordinator()
                threads = tf.train.start_queue_runners(coord=coord)
                # print("Finetune starting!")
                for epoch in range(config.training_epochs):
                    # sae.current_step = epoch
                    # start_time = time.time()
                    avg_cost = 0.
                    avg_cost1 = 0.
                    total_batch = int(config.n_samples / config.batch_size)
                    for x in range(total_batch):
                        batch_xs = get_random_block_from_data(plane_image, config.batch_size)
                        cost, _, cost1, _ = sess.run((sae0.cost, sae0.optimizer, sae1.cost, sae1.optimizer),
                                                     feed_dict={image: batch_xs})

                        avg_cost += cost / config.n_samples * config.batch_size
                        avg_cost1 += cost1 / config.n_samples * config.batch_size
                    if (avg_cost < 0.00004) and (avg_cost1 < 0.00004):
                        print("Epoch:", '%d,' % (epoch + 1), "Cost:",
                              "{:.9f}".format(avg_cost), "Cost1:",
                              "{:.9f}".format(avg_cost1),
                              "total time is ", "{:.9f}".format(time.time() - start_time),
                              " batch_size is", config.batch_size,
                              " hidden_size is", config.hidden_size)
                        break
                    if epoch > 15000:
                        print("Epoch:", '%d,' % (epoch + 1), "Cost:",
                              "{:.9f}".format(avg_cost), "Cost1:",
                              "{:.9f}".format(avg_cost1),
                              "total time is ", "{:.9f}".format(time.time() - start_time),
                              " batch_size is", config.batch_size,
                              " hidden_size is", config.hidden_size)
                        break
                    # Display logs per epoch step
                    # if epoch % config.display_step == 0:
                    #     print("Epoch:", '%d,' % (epoch + 1),
                    #           "Cost1:", "{:.9f}".format(avg_cost),
                    #           "Cost2:", "{:.9f}".format(avg_cost1),
                    #           "Time/Epoch is ", (time.time() - start_time))

                # saver.save(sess, checkpoint_path)
                coord.request_stop()
                coord.join(threads)

# print ("End, total time is ", time.time()-start_time)

