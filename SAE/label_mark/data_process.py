# -*- coding: utf-8 -*-
# 20180419
# 使用鼠标点击标签像素进行标签制作
import numpy as np
import os
import cv2
import scipy.io as sio
import matplotlib.pyplot as plt
from read_data import write_txt, read_txt
filename = os.path.join(os.getcwd(), 'hyper_data/ship_losangeles')
load_full_image = filename + '.mat'
label_file = filename + '_label.mat'

plane = sio.loadmat(load_full_image)
trX = plane['image']

image = np.zeros([trX.shape[0], trX.shape[1], 3], dtype=np.int16)
image[:, :, 1] = trX[:, :, 11]
image[:, :, 0] = trX[:, :, 19]

image[:, :, 2] = trX[:, :, 28]

plt.imshow(trX[:, :, 19])
plt.show()
# sio.savemat("ship_maxico_part2.mat", {'image': trX})  # 同理，只是存入了两个不同的变量供使用

# plane_new = sio.loadmat('ship_maxico_part2.mat')
# trY = plane_new['image']
# print np.sum(trX-trY)


