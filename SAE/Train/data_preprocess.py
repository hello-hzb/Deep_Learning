# -*- coding: utf-8 -*-
import numpy as np
import scipy.io as sio
import tensorflow as tf


def get_random_block_from_data(data, batch_size):
    start_index = np.random.randint(0, len(data) - batch_size)  # 产生0到（len(data) - batch_size）的随机数
    return data[start_index:(start_index + batch_size)]         # 取data start_index开始的batch_size个数


def read_hyperimage(dir_filename):
    load_data = sio.loadmat(dir_filename)
    load_matrix = load_data['or_image']   # 根据不同的数据，需要改变一下tuple的名字

    row = load_matrix.shape[0]
    column = load_matrix.shape[1]
    channel = load_matrix.shape[2]

    facet = load_matrix.reshape(row*column, channel)
    facet = np.array(facet, dtype=np.float32)
    trx = facet/facet.max()
    return trx


def OneWindow_Edge_Sel(OrImg, w, l, i, j):
    w2 = int(w/2)
    l2 = int(l/2)
    Temp = np.zeros((2*w+1, 2*l+1, OrImg.shape[2]))
    Temp[:, :, :] = OrImg[i-w:i+w+1, j-l:j+l+1, :]
    train_dataset = np.zeros((4 * (w + l) + 1, OrImg.shape[2]))
    # train_dataset = zeros(2*(w+l)+1,size(OrImg,3));
    train_dataset[0:2*w+1, :] = Temp[0, :, :]
    train_dataset[2*w+1:2*w+1+2*w+1, :] = Temp[Temp.shape[0]-1, :, :]
    train_dataset[4*w+2:4*w+1+2*l, :] = Temp[1:Temp.shape[0]-1, 0, :]
    train_dataset[4*w+1+2*l:4*w+1+2*l + 2*l-1, :] = Temp[1:Temp.shape[0]-1, Temp.shape[1]-1, :]
    train_dataset[4*w+1+2*l + 2*l-1, :] = OrImg[i, j, :]
    return train_dataset


def save_variable_to_h(name, data, dtype='float'):
    def out1d(vec):  # print an array with appropriate separator
        str = ',\n'.join(map(lambda x: ('%.9e' % x), vec))    # 保留9位小数,每一个数后面加上 ,\n
        return str

    if data.ndim == 1:  # 如果是一维数组的话 保存方式
        ni = len(data)
        p = '%s %s[%d]' % (dtype, name, ni)  # 保存格式数据类型+空格+变量名+[+数据个数+]
        data2str = out1d(data)
    else:
        (ni, nj) = data.shape
        p = '%s %s[%d]' % (dtype, name, ni*nj)
        data2str = ',\n'.join(map(out1d, [data[i, :] for i in range(ni)]))

    pa = ' = {'
    s = p + pa + data2str + '};\n'
    return s
