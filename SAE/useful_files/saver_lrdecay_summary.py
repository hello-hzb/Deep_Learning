# -*- coding: utf-8 -*-
# 通过tf.train.saver来保存模型,
import tensorflow as tf
import numpy as np
import os
train_dir = os.path.join(os.getcwd(), 'log_dir_gpu/')

# 使用 NumPy 生成假数据(phony data), 总共 100 个点.
x_data = np.float32(np.random.rand(2, 100))  # 随机输入
y_data = np.dot([0.100, 0.200], x_data) + 0.300

aa = tf.placeholder(dtype=tf.float32, shape=(2, 100), name='aa')
b = tf.Variable(tf.zeros([1]), name="bias")
W = tf.Variable(tf.random_uniform([1, 2], -1.0, 1.0))
y = tf.matmul(W, aa) + b

loss = tf.reduce_mean(tf.square(y - y_data))
###################################################################
lr_init = 0.5         # 初始学习率
decay_steps = 500     # 每10步衰减一次
decay_factor = 0.9    # 学习率衰减率
# current_step 在学习的过程中要根据训练的步数进行重新赋值，所以设置为variable
current_step = tf.get_variable('current_step', [], initializer=tf.constant_initializer(0), trainable=False)
lr = tf.train.exponential_decay(lr_init, current_step, decay_steps, decay_factor, staircase=True)
tf.summary.scalar(name="learning rate", tensor=lr)

optimizer = tf.train.GradientDescentOptimizer(lr)
train = optimizer.minimize(loss, global_step=current_step)  # 变学习率设置后需要在此处将current_step传入


tf.summary.histogram(name="bias", values=b)
tf.summary.histogram(name="weight", values=W)
# tf.summary.scalar(name="loss", tensor=loss)

saver = tf.train.Saver(tf.global_variables())
merged = tf.summary.merge_all()  # 将所有需要保存的数据进行合并
# 初始化变量
init = tf.global_variables_initializer()

# 启动图 (graph)
with tf.Session() as sess:
    sess.run(init)
    summary_writer = tf.summary.FileWriter(train_dir, sess.graph)  # 创建图写入器并写文件
    # 拟合平面
    for step in xrange(0, 10000):
        current_step = step                             # 在此处对根据学习的轮数对current_step进行赋值
        sess.run(train, feed_dict={aa: x_data})         # 训练
        summary_str = sess.run(merged)                  # 会话运行summary获得数据
        summary_writer.add_summary(summary_str, step)   # 将获得的数据加入到summary_writer写到tensorboard
        if step % 20 == 0:
            print step, sess.run(W), sess.run(b), sess.run(lr)

    checkpoint_path = os.path.join(train_dir, 'model.ckpt')
    saver.save(sess, checkpoint_path)
