# -*- coding: utf-8 -*-
# 20180419
# 使用鼠标点击标签像素进行标签制作
import numpy as np
import os
import cv2
import scipy.io as sio
import matplotlib.pyplot as plt
from read_data import write_txt, read_txt

# ship_losangeles
# ship_maxico_part1
# ship_maxico_part2

# filename = os.path.join(os.getcwd(), 'hyper_data/ship_losangeles')
filename = os.path.join(os.getcwd(), 'hyper_data/maxicoAA')

load_full_image = filename + '.mat'
label_file = filename + '_label.mat'

plane = sio.loadmat(load_full_image)
trX = plane['AA']
# 输入字符确定是否从文件总读取之前标注的文件
str = raw_input("load previous file?:[y]/n")
if str == 'n':
    label_mask = np.zeros([trX.shape[0], trX.shape[1]], dtype=np.int8)
else:
    # label_new = sio.loadmat(label_file)
    label_mask = plane['labelAA']

# sio.savemat(label_file, {'label': label_mask})  # 同理，只是存入了两个不同的变量供使用


def on_key_press(event):
    if event.key in 'm':                                                          # 输入m开始进行标签坐标识别
        print event.key
        pos = plt.ginput(5)                                                      # 手动点击图片上的像素点，获得目标位置
        # print pos
        for coor in pos:
            coordinate = np.array(coor)    # 将上一步得到的list转化成ndarray
            coordinate = np.array(np.floor((coordinate + 0.5)), dtype=np.int16)   # 由于坐标从-0.5开始且不是整数，故作取整处理
            label_mask[coordinate[1], coordinate[0]] = 1
        ax1.imshow(label_mask)

    if event.key in 'd':                                                          # 输入d开始删除标签像素
        print event.key
        pos = plt.ginput(1)                                                       # 手动点击图片上的像素点，获得目标位置
        print pos
        for coor in pos:
            coordinate = np.array(coor)                                          # 将上一步得到的list转化成ndarray
            coordinate = np.array(np.floor((coordinate + 0.5)), dtype=np.int16)  # 由于坐标从-0.5开始且不是整数，故作取整处理
            label_mask[coordinate[1], coordinate[0]] = 0
        ax1.imshow(label_mask)

    if event.key in 's':                                                         # 输入s保存label到txt中
        sio.savemat(load_full_image, {'labelAA': label_mask,'AA':trX})  # 同理，只是存入了两个不同的变量供使用
    sio.savemat(load_full_image, {'labelAA': label_mask,'AA':trX})  # 同理，只是存入了两个不同的变量供使用
    fig.canvas.draw_idle()
    # 重新绘制整个图表


# 绘制一个能够显示原图和标签图的plot
fig, ax = plt.subplots(1, 2)
ax0, ax1 = ax.ravel()
ax0.imshow((trX[:, :, 12])) # 左边的图显示原图
ax1.imshow(label_mask)        # 右边的图显示标签图

fig.canvas.mpl_disconnect(fig.canvas.manager.key_press_handler_id)  # 取消默认快捷键的注册
fig.canvas.mpl_connect('key_press_event', on_key_press)
plt.show()





# write_txt(label_file, label_mask)



# def on_key_press(event):
#     if event.key in 'rgbcmyk':
#         line.set_color(event.key)
#         print event.key
#     fig.canvas.draw_idle()#重新绘制整个图表，
#
# fig, ax = plt.subplots()
# x = np.linspace(0, 10, 1000)
# y = np.sin(x)
# line = ax.plot(x, y)[0]
#
# fig.canvas.mpl_disconnect(fig.canvas.manager.key_press_handler_id)#取消默认快捷键的注册
# fig.canvas.mpl_connect('key_press_event', on_key_press)
# plt.show()