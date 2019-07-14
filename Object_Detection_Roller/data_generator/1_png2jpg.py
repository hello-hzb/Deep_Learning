# -*- coding:utf-8 -*-

import os
import cv2
import sys
import numpy as np
path = "/home/ubuntu/SSD/data_generator/images_ly2/"
path2 = "/home/ubuntu/SSD/data_generator/ly/"

print(path)
for filename in os.listdir(path):
    if os.path.splitext(filename)[1] == '.png':
        # print(filename)
        img = cv2.imread(path + filename)
        print(filename.replace(".png", ".jpg"))
        newfilename = filename.replace(".png", ".jpg")
        # cv2.imshow("Image",img)
        # cv2.waitKey(0)
        cv2.imwrite(path2 + "A" + newfilename, img)


# """
#     先来说一下jpg图片和png图片的区别
#     jpg格式:是有损图片压缩类型,可用最少的磁盘空间得到较好的图像质量
#     png格式:不是压缩性,能保存透明等图
# """
# from PIL import Image
# import cv2 as cv
# import os
#
#
# def PNG_JPG(PngPath):
#     img = cv.imread(PngPath, 0)
#     w, h = img.shape[::-1]
#     infile = PngPath
#     outfile = os.path.splitext(infile)[0] + ".jpg"
#     img = Image.open(infile)
#     img = img.resize((int(w / 2), int(h / 2)), Image.ANTIALIAS)
#     try:
#         if len(img.split()) == 4:
#             # prevent IOError: cannot write mode RGBA as BMP
#             r, g, b, a = img.split()
#             img = Image.merge("RGB", (r, g, b))
#             img.convert('RGB').save(outfile, quality=70)
#             os.remove(PngPath)
#         else:
#             img.convert('RGB').save(outfile, quality=70)
#             os.remove(PngPath)
#         return outfile
#     except Exception as e:
#         print("PNG转换JPG 错误", e)
#
#
# if __name__ == '__main__':
#     files = os.listdir("/home/ubuntu/SSD/data_generator/images_ly/")
#     # 对文件名清单里的每一个文件名进行处理
#     for filename in files:
#         PNG_JPG("/home/ubuntu/SSD/data_generator/images_ly/"+filename)

