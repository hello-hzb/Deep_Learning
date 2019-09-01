# -*- coding: utf-8 -*-
# 2017/11/14
# roc,auc计算及绘图，输入标签及预测值即可
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve, auc
import matplotlib.pyplot as plt

# import matplotlib as mpl
from matplotlib.font_manager import FontProperties
plt.rcParams['font.sans-serif'] = ['NSimSun'] # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False    # 用来正常显示负号
font = FontProperties(fname="/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc", size=13)


y = np.array([0, 0, 1, 1])
scores = np.array([0.1, 0.7, 0.9, 0.9])

fpr, tpr, thresholds = roc_curve(y, scores)   # 输出数据维度和阈值有关，阈值的取值和预测值有关，等于预测值

roc_auc = roc_auc_score(y, scores)            # 计算AUC值
print roc_auc

##############################################################################
# Plot of a ROC curve for a specific class 将结果绘图输出
plt.figure()
lw = 2
plt.plot(fpr, tpr, color='darkorange', lw=lw, label='ROC (AUC = %0.4f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')


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
