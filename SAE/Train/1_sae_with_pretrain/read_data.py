# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import scipy.io as sio
import xlrd
import xlwt
import datetime

data_dir = '/home/ubuntu/my_file/hyper_sae/hyper_data/'

'''
##############################################
# 读取高光谱图像数据
load_fn = '../hyper_data/plane.mat'
# load_data = sio.loadmat(load_fn)
load_matrix = load_data['y']   # 100*100*126
facet = load_matrix.reshape(10000, 126)
facet = np.array(facet, dtype=np.float32)
trX = facet / facet.max()
trRef = trX    # float32
##############################################
'''


def read_tif(dir_filename):
    print("Read tif image! File name is", dir_filename)
    return


def write_tif(dir_filename):
    print("Write tif image! File name is ", dir_filename)
    return


def read_excel(dir_filename):  # 未完成，后续再写
    print ('Read excel! File name is ', dir_filename)

    print("It is ")
    return


def write_excel(dir_filename, data):  # 未完成，后续细化内容
    print ("Write excle! File name is ", dir_filename)
    style0 = xlwt.easyxf('font: name Times New Roman, color-index red, bold on', num_format_str='#,##0.00')
    style1 = xlwt.easyxf(num_format_str='D-MMM-YY')

    wb = xlwt.Workbook()
    ws = wb.add_sheet('A Test Sheet')

    ws.write(0, 0, 1234.56, style0)
    ws.write(1, 0, datetime.now(), style1)
    ws.write(2, 0, 1)
    ws.write(2, 1, 1)
    ws.write(2, 2, xlwt.Formula("A3+B3"))
    wb.save('example.xls')
    return

import scipy.io as sio

def read_mat(dir_filename):
    print ("Read mat! File name is", dir_filename)
    load_data = sio.loadmat(dir_filename)
    print ("It is done!")
    return load_data


def write_mat(dir_filename, data):
    print ("Write mat! File name is", dir_filename)
    sio.savemat(dir_filename, {'array': data})
    # sio.savemat(dir_filename, {'array1': data1, 'array2': data2})  # 同理，只是存入了两个不同的变量供使用
    print ("It is done!")
    return


def write_txt(dir_filename, data):
    print ("Write txt! File name is", dir_filename)
    np.savetxt(dir_filename, data)
    print ("It is done!")
    return


def read_txt(dir_filename):
    print ("Read txt! File name is ", dir)
    load_data = np.loadtxt(dir_filename)
    print ("It is done!")
    return load_data


def normalize_data(input_data):
    output_data = input_data/input_data.max()
    return output_data
