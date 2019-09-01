# -*- coding: utf-8 -*-
# 20180526
#
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

# 1 读取高光谱图像数据
# 测试用的plane数据
load_ship = os.path.join(os.getcwd(), '../hyper_data/ship_losangeles.mat')
load_ship_label = os.path.join(os.getcwd(), '../hyper_data/ship_losangeles_label.mat')

ship = sio.loadmat(load_ship)
label = sio.loadmat(load_ship_label)

ship_image = ship['image']
trX = (ship_image.reshape(1818680, 224))/np.max(ship_image)
load_label = label['label'].reshape(1818680)/100

# sio.savemat("ship_image_label", {'image': trX, 'label': load_label})  # 同理，只是存入了两个不同的变量供使用
ship_rxd_score = sio.loadmat("/home/ubuntu-server/my_file/master_project/SAE_AD_20180427/7_matlab_work/ship_score.mat")
rxd_score = ship_rxd_score["ship_score"]
# plt.imshow(np.reshape(rxd_score, (100, 100)))
# plt.show()
roc_auc1 = roc_auc_score(load_label, rxd_score)
print(" RXD AUC is ", roc_auc1)
