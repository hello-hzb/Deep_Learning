# -*- coding:utf-8 -*-
import os

# 获取目录下文件名清单
files = os.listdir("/home/ubuntu/SSD/data_generator/labels_ly")
# 对文件名清单里的每一个文件名进行处理
for filename in files:
    portion = os.path.splitext(filename)  # portion为名称和后缀分离后的列表
    newname = "A" + portion[0] + ".xml"
    print(filename)  # 打印出要更改的文件名
    os.chdir("/home/ubuntu/SSD/data_generator/labels_ly")  # 修改工作路径
    os.rename(filename, newname)
