# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
import os
import math
log_dir = os.path.join(os.getcwd(), 'log_dir_v1/')


# 20171120
# 官方的定义方式
class Autoencoder(object):

    def __init__(self, n_input, n_hidden, transfer_function=tf.nn.softplus, optimizer=tf.train.AdamOptimizer()):
        self.n_input = n_input
        self.n_hidden = n_hidden
        self.transfer = transfer_function

        network_weights = self._initialize_weights()
        self.weights = network_weights

        # model
        self.x = tf.placeholder(tf.float32, [None, self.n_input])
        self.hidden = self.transfer(tf.add(tf.matmul(self.x, self.weights['w1']), self.weights['b1']))
        self.reconstruction = tf.add(tf.matmul(self.hidden, self.weights['w2']), self.weights['b2'])

        # cost
        self.cost = 0.5 * tf.reduce_sum(tf.pow(tf.subtract(self.reconstruction, self.x), 2.0))
        self.optimizer = optimizer.minimize(self.cost)

        init = tf.global_variables_initializer()
        self.sess = tf.Session()
        self.sess.run(init)

    def _initialize_weights(self):
        all_weights = dict()
        all_weights['w1'] = tf.get_variable("w1", shape=[self.n_input, self.n_hidden],
            initializer=tf.contrib.layers.xavier_initializer())
        all_weights['b1'] = tf.Variable(tf.zeros([self.n_hidden], dtype=tf.float32))
        all_weights['w2'] = tf.Variable(tf.zeros([self.n_hidden, self.n_input], dtype=tf.float32))
        all_weights['b2'] = tf.Variable(tf.zeros([self.n_input], dtype=tf.float32))
        return all_weights

    def partial_fit(self, X):
        cost, opt = self.sess.run((self.cost, self.optimizer), feed_dict={self.x: X})
        return cost

    def calc_total_cost(self, X):
        return self.sess.run(self.cost, feed_dict={self.x: X})

    def transform(self, X):
        return self.sess.run(self.hidden, feed_dict={self.x: X})

    def generate(self, hidden = None):
        if hidden is None:
            hidden = self.sess.run(tf.random_normal([1, self.n_hidden]))
        return self.sess.run(self.reconstruction, feed_dict={self.hidden: hidden})

    def reconstruct(self, X):
        return self.sess.run(self.reconstruction, feed_dict={self.x: X})

    def getWeights(self):
        return self.sess.run(self.weights['w1'])

    def getBiases(self):
        return self.sess.run(self.weights['b1'])


# 20171118
# 会话启动不在类中定义， 更新标志位所需要的计算不在类中
class Autoencoder2(object):

    def __init__(self, n_input, n_hidden, batch_size, transfer_function=tf.nn.softplus, optimizer=tf.train.AdamOptimizer()):
        self.n_input = n_input
        self.n_hidden = n_hidden
        self.transfer = transfer_function
        self.batch_size = batch_size

        # model
        network_weights = self._initialize_weights()
        self.weights = network_weights
        # with tf.device('/gpu:1'):
        self.x = tf.placeholder(tf.float32, [None, self.n_input])
        self.hidden = self.transfer(tf.add(tf.matmul(self.x, self.weights['w1']), self.weights['b1']))
        self.reconstruction = tf.add(tf.matmul(self.hidden, self.weights['w2']), self.weights['b2'])
        self.errtmp = tf.reduce_sum(tf.square(self.reconstruction - self.x), 1)
        # cost
        self.cost = 0.5 * tf.reduce_sum(tf.pow(tf.subtract(self.reconstruction, self.x), 2.0))
        self.optimizer = optimizer.minimize(self.cost)
        ############COMPUTE DISTANCE #################
        filter_win = tf.ones((batch_size, self.n_hidden))
        self.filter_win = tf.multiply(filter_win,  self.hidden[batch_size-1, :])
        self.Retmpdist = tf.reduce_sum(tf.square(self.hidden - self.filter_win), 1) / self.errtmp
        self.dis_err = tf.reduce_sum(self.Retmpdist)

    def _initialize_weights(self):
        all_weights = dict()
        #all_weights['w1'] = tf.Variable(tf.zeros([self.n_input, self.n_hidden], dtype=tf.float32))
        all_weights['w1'] = tf.get_variable("w1", shape=[self.n_input, self.n_hidden],
                                            initializer=tf.contrib.layers.xavier_initializer())
        all_weights['b1'] = tf.Variable(tf.zeros([self.n_hidden], dtype=tf.float32))
        all_weights['w2'] = tf.Variable(tf.zeros([self.n_hidden, self.n_input], dtype=tf.float32))
        all_weights['b2'] = tf.Variable(tf.zeros([self.n_input], dtype=tf.float32))
        return all_weights

    def partial_fit(self, X, sess):
        cost, opt = sess.run((self.cost, self.optimizer), feed_dict={self.x: X})
        return cost

    def calc_total_cost(self, X, sess):
        return sess.run(self.cost, feed_dict={self.x: X})

    def transform(self, X, sess):
        return sess.run(self.hidden, feed_dict={self.x: X})

    def generate(self, sess, hidden=None):
        if hidden is None:
            hidden = sess.run(tf.random_normal([1, self.n_hidden]))
        return sess.run(self.reconstruction, feed_dict={self.hidden: hidden})

    def reconstruct(self, X, sess):
        return sess.run(self.reconstruction, feed_dict={self.x: X})

    def getWeights(self, sess):
        return sess.run(self.weights['w1'])

    def getBiases(self, sess):
        return sess.run(self.weights['b1'])

    def Code_Reconstrct(self,X, sess):
        return sess.run((self.hidden,self.errtmp), feed_dict={self.x: X})

    def DistLocal(self, X, sess):
        return sess.run(self.dis_err, feed_dict={self.x: X})

    def Differe(self, X, sess):
        return sess.run((self.hidden, self.filter_win), feed_dict={self.x: X})


# 20171121
# 会话启动不在类中定义， 增加距离计算以及目标库计算，
class Autoencoder3(object):
    def __init__(self, n_input, n_hidden, batch_size, transfer_function=tf.nn.softplus,
                 optimizer=tf.train.AdamOptimizer()):
        self.n_input = n_input
        self.n_hidden = n_hidden
        self.transfer = transfer_function
        self.batch_size = batch_size

        # model
        network_weights = self._initialize_weights()
        self.weights = network_weights
        # with tf.device('/gpu:1'):
        self.x = tf.placeholder(tf.float32, [None, self.n_input])
        self.hidden = self.transfer(tf.add(tf.matmul(self.x, self.weights['w1']), self.weights['b1']))
        self.reconstruction = tf.add(tf.matmul(self.hidden, self.weights['w2']), self.weights['b2'])

        # cost
        self.cost = 0.5 * tf.reduce_sum(tf.pow(tf.subtract(self.reconstruction, self.x), 2.0))
        self.optimizer = optimizer.minimize(self.cost)
        # ###########COMPUTE DISTANCE #################
        self.errtmp = tf.reduce_sum(tf.square(self.reconstruction - self.x), 1)
        filter_win = tf.ones((batch_size, self.n_hidden))
        self.filter_win = tf.multiply(filter_win, self.hidden[batch_size - 1, :])
        self.Retmpdist = tf.reduce_sum(tf.square(self.hidden - self.filter_win), 1) / self.errtmp
        self.dis_err = tf.reduce_sum(self.Retmpdist)

        # 计算库更新所需要的操作
        self.dis_lib = tf.placeholder(tf.float32, [None, self.n_hidden])
        dis_ones = tf.ones(tf.shape(self.dis_lib))
        disLibtmp = tf.multiply(dis_ones, self.hidden[self.batch_size - 1, :])
        self.difLib = self.dis_lib - disLibtmp
        self.DisLib_t = tf.reduce_sum(tf.square(self.difLib))

    def _initialize_weights(self):
        all_weights = dict()
        # all_weights['w1'] = tf.Variable(tf.zeros([self.n_input, self.n_hidden], dtype=tf.float32))
        all_weights['w1'] = tf.get_variable("w1", shape=[self.n_input, self.n_hidden],
                                            initializer=tf.contrib.layers.xavier_initializer())
        all_weights['b1'] = tf.Variable(tf.zeros([self.n_hidden], dtype=tf.float32))
        all_weights['w2'] = tf.Variable(tf.zeros([self.n_hidden, self.n_input], dtype=tf.float32))
        all_weights['b2'] = tf.Variable(tf.zeros([self.n_input], dtype=tf.float32))
        return all_weights

    def partial_fit(self, X, sess):
        cost, opt = sess.run((self.cost, self.optimizer), feed_dict={self.x: X})
        return cost

    def calc_total_cost(self, X, sess):
        return sess.run(self.cost, feed_dict={self.x: X})

    def transform(self, X, sess):
        return sess.run(self.hidden, feed_dict={self.x: X})

    def generate(self, sess, hidden=None):
        if hidden is None:
            hidden = sess.run(tf.random_normal([1, self.n_hidden]))
        return sess.run(self.reconstruction, feed_dict={self.hidden: hidden})

    def reconstruct(self, X, sess):
        return sess.run(self.reconstruction, feed_dict={self.x: X})

    def getWeights(self, sess):
        return sess.run(self.weights['w1'])

    def getBiases(self, sess):
        return sess.run(self.weights['b1'])

    def Code_Reconstrct(self, X, sess):
        return sess.run((self.hidden, self.errtmp), feed_dict={self.x: X})

    def DistLocal(self, X, sess):
        return sess.run(self.dis_err, feed_dict={self.x: X})

    def Differe(self, X, sess):
        return sess.run((self.hidden, self.filter_win), feed_dict={self.x: X})

# 20171121
# 两种参数定义方式，仅供参考使用
class Autoencoder4(object):
    def __init__(self, n_input, n_hidden, transfer_function=tf.nn.softplus, optimizer=tf.train.AdamOptimizer()):
        self.n_input = n_input
        self.n_hidden = n_hidden
        self.transfer = transfer_function
        network_weights = self.weight_declare()
        self.weights = network_weights

        # model
        self.x = tf.placeholder(tf.float32, [None, self.n_input])
        self.hidden = self.transfer(tf.add(tf.matmul(self.x, self.weights['w1']), self.weights['b1']))

        self.reconstruction = tf.add(tf.matmul(self.hidden, self.weights['w2']), self.weights['b2'])

        # cost
        self.cost = 0.5 * tf.reduce_sum(tf.pow(tf.subtract(self.reconstruction, self.x), 2.0))
        self.optimizer = optimizer.minimize(self.cost)

        # init = tf.global_variables_initializer()

    def _initialize_weights(self):
        all_weights = dict()
        all_weights['w1'] = tf.Variable(tf.truncated_normal([self.n_input, self.n_hidden], mean=0.0, stddev=1.0, dtype=tf.float32), name="Weight1")
        all_weights['b1'] = tf.Variable(tf.truncated_normal([self.n_hidden], mean=0.0, stddev=1.0, dtype=tf.float32), name="Bias1")
        all_weights['w2'] = tf.Variable(tf.truncated_normal([self.n_hidden, self.n_input], dtype=tf.float32), name="Weight2")
        all_weights['b2'] = tf.Variable(tf.truncated_normal([self.n_input], dtype=tf.float32), name="Bias2")
        return all_weights

    def weight_declare(self):
        all_weights = dict()
        all_weights['w1'] = tf.get_variable(name="Weight1", initializer=tf.truncated_normal([self.n_input, self.n_hidden], mean=0.0, stddev=1.0, dtype=tf.float32))
        all_weights['b1'] = tf.get_variable(name="Bias1", initializer=tf.truncated_normal([self.n_hidden], mean=0.0, stddev=1.0, dtype=tf.float32))
        all_weights['w2'] = tf.get_variable(name="Weight2", initializer=tf.truncated_normal([self.n_hidden, self.n_input], dtype=tf.float32),)
        all_weights['b2'] = tf.get_variable(name="Bias2", initializer=tf.truncated_normal([self.n_input], dtype=tf.float32))
        return all_weights

    def partial_fit(self, session, X):
        cost, opt = session.run((self.cost, self.optimizer), feed_dict={self.x: X})
        return cost

    def calc_total_cost(self, session, X):
        return session.run(self.cost, feed_dict={self.x: X})

    def transform(self, session, X):
        return session.run(self.hidden, feed_dict={self.x: X})

    def generate(self, session, hidden=None):
        if hidden is None:
            hidden = session.run(tf.random_normal([1, self.n_hidden]))
        return session.run(self.reconstruction, feed_dict={self.hidden: hidden})

    def reconstruct(self, session, X):
        return session.run(self.reconstruction, feed_dict={self.x: X})

    def getWeights(self, session):
        return session.run(self.weights['w1'])

    def getBiases(self, session):
        return session.run(self.weights['b1'])


# 20171121 *****************************************************************************
# 以下3个类是类Autoencoder3的分解版本

# 构建单隐藏层的ae网络
class Autoencoderpure(object):
    def __init__(self, n_input, n_hidden, transfer_function=tf.nn.softplus,
                 optimizer=tf.train.AdamOptimizer()):
        self.n_input = n_input
        self.n_hidden = n_hidden
        self.transfer = transfer_function

        # model
        network_weights = self.get_variable_declare()
        self.weights = network_weights
        # with tf.device('/gpu:1'):
        self.x = tf.placeholder(tf.float32, [None, self.n_input])
        self.hidden = self.transfer(tf.add(tf.matmul(self.x, self.weights['w1']), self.weights['b1']))
        self.reconstruction = tf.add(tf.matmul(self.hidden, self.weights['w2']), self.weights['b2'])
        # cost
        self.cost = 0.5 * tf.reduce_mean(tf.square(self.reconstruction - self.x))
        self.errtmp = tf.reduce_sum(tf.square(self.reconstruction - self.x), axis=1, name='reconstruct_err')

        self.optimizer = optimizer.minimize(self.cost)

    def _initialize_weights(self):
        all_weights = dict()
        # all_weights['w1'] = tf.Variable(tf.zeros([self.n_input, self.n_hidden], dtype=tf.float32))
        all_weights['w1'] = tf.get_variable("w1", shape=[self.n_input, self.n_hidden],
                                            initializer=tf.contrib.layers.xavier_initializer())
        all_weights['b1'] = tf.Variable(tf.zeros([self.n_hidden], dtype=tf.float32))
        all_weights['w2'] = tf.Variable(tf.zeros([self.n_hidden, self.n_input], dtype=tf.float32))
        all_weights['b2'] = tf.Variable(tf.zeros([self.n_input], dtype=tf.float32))
        return all_weights

    def variable_declare(self):
        all_weights = dict()
        all_weights['w1'] = tf.Variable(tf.truncated_normal([self.n_input, self.n_hidden], mean=0.0, stddev=1.0, dtype=tf.float32), name="Weight1")
        all_weights['b1'] = tf.Variable(tf.truncated_normal([self.n_hidden], mean=0.0, stddev=1.0, dtype=tf.float32), name="Bias1")
        all_weights['w2'] = tf.Variable(tf.truncated_normal([self.n_hidden, self.n_input], dtype=tf.float32), name="Weight2")
        all_weights['b2'] = tf.Variable(tf.truncated_normal([self.n_input], dtype=tf.float32), name="Bias2")
        return all_weights

    def get_variable_declare(self):
        all_weights = dict()
        all_weights['w1'] = tf.get_variable(name="Weight1", initializer=tf.truncated_normal([self.n_input, self.n_hidden], mean=0.0, stddev=1.0, dtype=tf.float32))
        all_weights['b1'] = tf.get_variable(name="Bias1", initializer=tf.truncated_normal([self.n_hidden], mean=0.0, stddev=1.0, dtype=tf.float32))
        all_weights['w2'] = tf.get_variable(name="Weight2", initializer=tf.truncated_normal([self.n_hidden, self.n_input], dtype=tf.float32),)
        all_weights['b2'] = tf.get_variable(name="Bias2", initializer=tf.truncated_normal([self.n_input], dtype=tf.float32))
        return all_weights

    def partial_fit(self, X, sess):
        cost, opt = sess.run((self.cost, self.optimizer), feed_dict={self.x: X})
        return cost

    def calc_total_cost(self, X, sess):
        return sess.run(self.cost, feed_dict={self.x: X})

    def transform(self, X, sess):
        return sess.run(self.hidden, feed_dict={self.x: X})

    def generate(self, sess, hidden=None):
        if hidden is None:
            hidden = sess.run(tf.random_normal([1, self.n_hidden]))
        return sess.run(self.reconstruction, feed_dict={self.hidden: hidden})

    def reconstruct(self, X, sess):
        return sess.run(self.reconstruction, feed_dict={self.x: X})

    def getWeights(self, sess):
        return sess.run(self.weights['w1'])

    def getBiases(self, sess):
        return sess.run(self.weights['b1'])


# 计算重建误差并重构图像

class Anomalydetect(object):
    def __init__(self, batch_size, hidden, err_temp):
        self.hidden = hidden
        self.errtmp = err_temp
        filter_win = tf.ones(tf.shape(self.hidden), name='AD_filter_ones')     # 创建全1矩阵，大小和coding数据一样
        self.filter_win = tf.multiply(filter_win, self.hidden[batch_size - 1, :], name='AD_filter_win_multop')   # 将全1矩阵的值替换为coding输出的最后一个点的值
        self.Retmpdist = tf.reduce_sum(tf.square(self.hidden - self.filter_win), 1) / self.errtmp  # 计算coding输出与指定点的欧式距离并处于重建误差
        self.dis_err = tf.reduce_sum(self.Retmpdist, name='AD_dis_err')    # 存在疑问，需要与师兄确认


# class Anomalydetect(object):
#     def __init__(self, batch_size, hidden, err_temp):
#         self.errtmp = err_temp
#         filter_win = tf.ones(tf.shape(hidden), name='AD_filter_ones')  # 创建全1矩阵，大小和coding数据一样
#         self.filter_win = tf.multiply(filter_win, hidden[batch_size - 1, :], name='AD_filter_win_multop')  # 将全1矩阵的值替换为coding输出的最后一个点的值
#         self.Retmpdist = tf.reduce_sum(tf.square(hidden - self.filter_win), 1) / self.errtmp  # 计算coding输出与指定点的欧式距离并处于重建误差
#         self.dis_err = tf.reduce_sum(self.Retmpdist, name='AD_dis_err')  # 存在疑问，需要与师兄确认


# 构建目标库需要的计算
class Libcal(object):
    def __init__(self, n_hidden, hidden):
        self.n_hidden = n_hidden
        self.hidden = hidden
        self.dis_lib = tf.placeholder(tf.float32, [None, None, self.n_hidden], name='dis_lib_input')
        dis_ones = tf.ones(tf.shape(self.dis_lib))
        # disLibtmp = tf.multiply(dis_ones, self.hidden[self.batch_size - 1, :])
        self.disLibtmp = tf.multiply(dis_ones, self.hidden[-1, :])
        self.difLib = self.dis_lib - self.disLibtmp
        self.DisLib_t = tf.reduce_mean(tf.square(self.difLib), name='lib_dis')

class UpdateF(object):
    def __init__(self, batch_size, hidden, n_hidden):
        self.batch_size = batch_size
        # hidden, differe, filter_win = (autoencoder.Differe(tmptrXLocal))
        with tf.name_scope("AD"):
            self.UpdatingFlag = tf.Variable(initial_value=tf.constant(False, dtype=tf.bool), trainable=False,
                                            name="updating_flag", dtype=tf.bool)

            self.hidden = hidden
            self.n_hidden = n_hidden
            self.DistLib = {"libcode": [], "distl": []}
            self.disliblen = tf.Variable(initial_value=tf.constant(0, dtype=tf.int32), trainable=False,
                                         dtype=tf.int32)
            self.ttttttt = 0
            self.DisLib_t = tf.Variable(initial_value=tf.constant(0.0, dtype=tf.float32), trainable=False,
                                        dtype=tf.float32)
            self.difLib = []

            self.upflag = tf.case([(tf.not_equal(self.disliblen, 0), self.f2)], default=self.f1)

    def f1(self):
        self.DistLib['libcode'].append(self.hidden[self.batch_size - 1])
        self.DistLib['distl'].append([0])
        self.disliblen += 1
        # self.UpdatingFlag = 0
        tf.assign(self.UpdatingFlag, tf.constant(False, dtype=tf.bool))
        self.ttttttt = 8
        return self.UpdatingFlag

    def f2(self):
        self.ttttttt = 12
        tf.assign(self.UpdatingFlag, tf.constant(False))
        DistLen = tf.ones((self.disliblen, self.n_hidden))
        disLibtmp = DistLen * self.hidden[self.batch_size - 1, :]
        self.difLib = self.DistLib['libcode'] - disLibtmp
        self.DisLib_t = tf.reduce_sum(tf.square(self.difLib)) / DistLen
        if self.DisLib_t > self.DistLib['dist1'][-1]:
            self.disliblen += 1
            self.DistLib['libcode'].append(self.hidden[self.batch_size - 1])
            self.DistLib['dist1'].append(self.DisLib_t)
            tf.assign(self.UpdatingFlag, tf.constant(True, dtype=tf.bool))
        return self.UpdatingFlag


# 该类利用类堆叠的方式构建网络时，预训练以后取各隐藏层创建网络的类
class SAE(object):
    def __init__(self,
                 sae_tuple,
                 stack_size,
                 hidden_size,
                 n_input,
                 transfer_function=tf.nn.softplus,
                 optimizer=tf.train.AdamOptimizer()):
        self.n_input = n_input
        self.transfer = transfer_function
        self.w_out = tf.get_variable(name="Weight_out", initializer=tf.truncated_normal([hidden_size[-1], self.n_input], mean=0.0, stddev=1.0, dtype=tf.float32))
        self.b_out = tf.get_variable(name="Bias_out", initializer=tf.truncated_normal([self.n_input], mean=0.0, stddev=1.0, dtype=tf.float32))

        # model
        self.x = tf.placeholder(tf.float32, [None, self.n_input], name='SAE_input')
        hidden = self.x
        for i in xrange(stack_size):
            with tf.name_scope('hidden_cal%d' % i):
                hidden = self.transfer(tf.add(tf.matmul(hidden, sae_tuple[i].weights['w1']), sae_tuple[i].weights['b1']))
                if math.ceil(stack_size/2) == i:
                    self.hidden_out = hidden

        self.reconstruction = tf.add(tf.matmul(hidden, self.w_out), self.b_out)
        # cost
        self.cost = 0.5 * tf.reduce_mean(tf.square(self.reconstruction - self.x))
        self.errtmp = tf.reduce_sum(tf.square(self.reconstruction - self.x), axis=1, name='reconstruct_err')
        self.optimizer = optimizer.minimize(self.cost)

    def partial_fit(self, X, sess):
        cost, opt = sess.run((self.cost, self.optimizer), feed_dict={self.x: X})
        return cost

    def calc_total_cost(self, X, sess):
        return sess.run(self.cost, feed_dict={self.x: X})

    def transform(self, X, sess):
        return sess.run(self.hidden_out, feed_dict={self.x: X})

    def reconstruct(self, X, sess):
        return sess.run(self.reconstruction, feed_dict={self.x: X})
# *****************************************************************************************


# 20171125 *****************************************************************************
# 以下3个类是类Autoencoderpure的无优化器版本
# 构建单隐藏层的ae网络
class Autoencodernopt(object):
    def __init__(self, n_input, n_hidden, transfer_function=tf.nn.softplus,
                 optimizer=tf.train.AdamOptimizer()):
        self.n_input = n_input
        self.n_hidden = n_hidden
        self.transfer = transfer_function

        # model
        network_weights = self.get_variable_declare()
        self.weights = network_weights
        # with tf.device('/gpu:1'):
        self.x = tf.placeholder(tf.float32, [None, self.n_input])
        self.hidden = self.transfer(tf.add(tf.matmul(self.x, self.weights['w1']), self.weights['b1']))
        self.reconstruction = tf.add(tf.matmul(self.hidden, self.weights['w2']), self.weights['b2'])
        # cost
        self.cost = 0.5 * tf.reduce_mean(tf.square(self.reconstruction - self.x))
        # self.optimizer = optimizer.minimize(self.cost)

    def get_variable_declare(self):
        all_weights = dict()
        all_weights['w1'] = tf.get_variable(name="Weight1", initializer=tf.truncated_normal([self.n_input, self.n_hidden], mean=0.0, stddev=1.0, dtype=tf.float32))
        all_weights['b1'] = tf.get_variable(name="Bias1", initializer=tf.truncated_normal([self.n_hidden], mean=0.0, stddev=1.0, dtype=tf.float32))
        all_weights['w2'] = tf.get_variable(name="Weight2", initializer=tf.truncated_normal([self.n_hidden, self.n_input], dtype=tf.float32),)
        all_weights['b2'] = tf.get_variable(name="Bias2", initializer=tf.truncated_normal([self.n_input], dtype=tf.float32))
        return all_weights

    def calc_total_cost(self, X, sess):
        return sess.run(self.cost, feed_dict={self.x: X})

    def transform(self, X, sess):
        return sess.run(self.hidden, feed_dict={self.x: X})

    def generate(self, sess, hidden=None):
        if hidden is None:
            hidden = sess.run(tf.random_normal([1, self.n_hidden]))
        return sess.run(self.reconstruction, feed_dict={self.hidden: hidden})

    def reconstruct(self, X, sess):
        return sess.run(self.reconstruction, feed_dict={self.x: X})

    def getWeights(self, sess):
        return sess.run(self.weights['w1'])

    def getBiases(self, sess):
        return sess.run(self.weights['b1'])


# 该类利用类堆叠的方式构建网络时，预训练以后取各隐藏层创建网络的类,只包含前向过程
class SAEnopt(object):
    def __init__(self,
                 sae_tuple,
                 stack_size,
                 hidden_size,
                 n_input,
                 transfer_function=tf.nn.softplus,
                 optimizer=tf.train.AdamOptimizer()):
        self.n_input = n_input
        self.transfer = transfer_function
        self.w_out = tf.get_variable(name="Weight_out", initializer=tf.truncated_normal([hidden_size[-1], self.n_input], mean=0.0, stddev=1.0, dtype=tf.float32))
        self.b_out = tf.get_variable(name="Bias_out", initializer=tf.truncated_normal([self.n_input], mean=0.0, stddev=1.0, dtype=tf.float32))

        # model
        self.x = tf.placeholder(tf.float32, [None, self.n_input], name='SAE_input')
        hidden = self.x
        for i in xrange(stack_size):
            with tf.name_scope('hidden_cal%d' % i):
                hidden = self.transfer(tf.add(tf.matmul(hidden, sae_tuple[i].weights['w1']), sae_tuple[i].weights['b1']))
                if math.ceil(stack_size/2) == i:
                    self.hidden_out = hidden

        self.reconstruction = tf.add(tf.matmul(hidden, self.w_out), self.b_out)
        # cost
        self.cost = 0.5 * tf.reduce_mean(tf.square(self.reconstruction - self.x))
        self.errtmp = tf.reduce_sum(tf.square(self.reconstruction - self.x), axis=1, name='reconstruct_err')
        # self.optimizer = optimizer.minimize(self.cost)

    def calc_total_cost(self, X, sess):
        return sess.run(self.cost, feed_dict={self.x: X})

    def transform(self, X, sess):
        return sess.run(self.hidden_out, feed_dict={self.x: X})

    def reconstruct(self, X, sess):
        return sess.run(self.reconstruction, feed_dict={self.x: X})

# *****************************************************************************************


# 20171127
# 1.能够自定义隐藏层层数，及节点数
# 2.输入数据不再是placeholder形式
# 3.会话不在类中定义
# 4.默认指定硬件设备为GPU
class SAEnoplh(object):
    def __init__(self,
                 input_data,
                 n_input,
                 stack_size,
                 hidden_size,
                 transfer_function=tf.nn.softplus,
                 optimizer=tf.train.AdamOptimizer()
                 ):
        self.weights = dict()
        # model
        hidden = input_data
        for i in xrange(stack_size):
            if i == 0:
                input_size = n_input
            else:
                input_size = hidden_size[i-1]

            with tf.variable_scope('hidden_cal%d' % i):
                self.weights['w%d' % i] = tf.get_variable(name=('weight%d' % i),
                                                          initializer=tf.truncated_normal(
                                                              [input_size, hidden_size[i]],
                                                              mean=0.0, stddev=1.0,
                                                              dtype=tf.float32))
                self.weights['b%d' % i] = tf.get_variable(name=('bias%d' % i),
                                                          initializer=tf.truncated_normal(
                                                              [hidden_size[i]],
                                                              mean=0.0,
                                                              stddev=1.0,
                                                              dtype=tf.float32))
                hidden = transfer_function(tf.add(tf.matmul(hidden, self.weights['w%d' % i]), self.weights['b%d' % i]))
                if math.ceil(stack_size/2) == i:
                    self.hidden_out = hidden
                    self.shape = tf.shape(self.hidden_out)

        with tf.variable_scope('output_layer'):
            self.w_out = tf.get_variable(name="Weight_out",
                                         initializer=tf.truncated_normal([hidden_size[stack_size-1], n_input], mean=0.0, stddev=1.0,
                                                                         dtype=tf.float32))
            self.b_out = tf.get_variable(name="Bias_out",
                                         initializer=tf.truncated_normal([n_input], mean=0.0, stddev=1.0, dtype=tf.float32))

            self.reconstruction = tf.add(tf.matmul(hidden, self.w_out), self.b_out)
        # cost
        with tf.name_scope("cost_errtmp"):
            # 和下面的写法效果是一样的，就是最后一步是求平均
            self.cost = 0.5 * tf.reduce_mean(tf.square(self.reconstruction - input_data), name='cost')
            # 和上面的写法还是一样的，就是最后一步是求和
            # self.cost = 0.5 * tf.reduce_sum(tf.pow(tf.subtract(self.reconstruction, input_data), 2.0))
            # axis=1对列求和
            self.errtmp = tf.reduce_sum(tf.square(self.reconstruction - input_data), axis=1, name='reconstruct_err')
        self.optimizer = optimizer.minimize(self.cost)

    def calc_total_cost(self, sess):
        return sess.run(self.cost)

    def transform(self, sess):
        return sess.run(self.hidden_out)

    def reconstruct(self, sess):
        return sess.run(self.reconstruction)


# *****************************************************************************************
# 20180131
# 1.能够自定义隐藏层层数，及节点数
# 2.输入数据不再是placeholder形式
# 3.会话不在类中定义
# 4.默认指定硬件设备为GPU
class SAEnoplhV2(object):
    def __init__(self,
                 input_data,
                 n_input,
                 stack_size,
                 hidden_size,
                 transfer_function=tf.nn.relu,
                 optimizer=tf.train.AdamOptimizer()
                 ):
        self.weights = dict()
        # model
        hidden = input_data
        for i in xrange(stack_size):
            if i == 0:
                input_size = n_input
            else:
                input_size = hidden_size[i-1]

            with tf.variable_scope('hidden_cal%d' % i):
                with tf.device('/cpu:0'):
                    self.weights['w%d' % i] = tf.get_variable(name=('weight%d' % i),
                                                              initializer=tf.truncated_normal(
                                                                  [input_size, hidden_size[i]],
                                                                  mean=0.0, stddev=1.0,
                                                                  dtype=tf.float32))
                    self.weights['b%d' % i] = tf.get_variable(name=('bias%d' % i),
                                                              initializer=tf.truncated_normal(
                                                                  [hidden_size[i]],
                                                                  mean=0.0,
                                                                  stddev=1.0,
                                                                  dtype=tf.float32))
                hidden = transfer_function(tf.add(tf.matmul(hidden, self.weights['w%d' % i]), self.weights['b%d' % i]))
                if math.ceil(stack_size/2) == i:
                    self.hidden_out = hidden
                    self.shape = tf.shape(self.hidden_out)

        with tf.variable_scope('output_layer'):
            with tf.device('/cpu:0'):
                self.w_out = tf.get_variable(name="Weight_out",
                                             initializer=tf.truncated_normal([hidden_size[stack_size-1], n_input], mean=0.0, stddev=1.0,
                                                                             dtype=tf.float32))
                self.b_out = tf.get_variable(name="Bias_out",
                                             initializer=tf.truncated_normal([n_input], mean=0.0, stddev=1.0, dtype=tf.float32))

            self.reconstruction = transfer_function(tf.add(tf.matmul(hidden, self.w_out), self.b_out))
        # cost
        with tf.name_scope("cost_errtmp"):
            # 和下面的写法效果是一样的，就是最后一步是求平均
            self.cost = 0.5 * tf.reduce_mean(tf.square(self.reconstruction - input_data), name='cost')
            # 和上面的写法还是一样的，就是最后一步是求和
            # self.cost = 0.5 * tf.reduce_sum(tf.pow(tf.subtract(self.reconstruction, input_data), 2.0))
            # axis=1对列求和
            self.errtmp = tf.square(self.reconstruction - input_data)
        self.optimizer = optimizer.minimize(self.cost)

    def calc_total_cost(self, sess):
        return sess.run(self.cost)

    def transform(self, sess):
        return sess.run(self.hidden_out)

    def reconstruct(self, sess):
        return sess.run(self.reconstruction)

# *****************************************************************************************


# *****************************************************************************************
# 20180131
# 1.能够自定义隐藏层层数，及节点数
# 2.输入数据不再是placeholder形式
# 3.会话不在类中定义
# 4.默认指定硬件设备为GPU
# 5.参数初始化方式和v2不一样
class SAEnoplhV4(object):
    def __init__(self,
                 input_data,
                 n_input,
                 stack_size,
                 hidden_size,
                 transfer_function=tf.nn.leaky_relu,
                 optimizer=tf.train.AdamOptimizer()
                 ):
        self.weights = dict()
        # model
        hidden = input_data
        for i in xrange(stack_size):
            if i == 0:
                input_size = n_input
            else:
                input_size = hidden_size[i-1]

            with tf.variable_scope('hidden_cal%d' % i):
                with tf.device('/cpu:0'):
                    self.weights['w%d' % i] = tf.get_variable(name=('weight%d' % i),
                                                              shape=[input_size, hidden_size[i]],
                                                              dtype=tf.float32)
                    self.weights['b%d' % i] = tf.get_variable(name=('bias%d' % i),
                                                              initializer=tf.zeros([hidden_size[i]], dtype=tf.float32))
                hidden = transfer_function(tf.add(tf.matmul(hidden, self.weights['w%d' % i]), self.weights['b%d' % i]))
                if math.ceil(stack_size/2) == i:
                    self.hidden_out = hidden
                    self.shape = tf.shape(self.hidden_out)

        with tf.variable_scope('output_layer'):
            with tf.device('/cpu:0'):
                self.w_out = tf.get_variable(name="Weight_out",
                                             shape=[hidden_size[stack_size-1], n_input],
                                             dtype=tf.float32)
                self.b_out = tf.get_variable(name="Bias_out",
                                             initializer=tf.zeros([n_input], dtype=tf.float32))

            self.reconstruction = transfer_function(tf.add(tf.matmul(hidden, self.w_out), self.b_out))
        # cost
        with tf.name_scope("cost_errtmp"):
            # 和下面的写法效果是一样的，就是最后一步是求平均
            self.cost = 0.5 * tf.reduce_mean(tf.square(self.reconstruction - input_data), name='cost')
            # 和上面的写法还是一样的，就是最后一步是求和
            # self.cost = 0.5 * tf.reduce_sum(tf.pow(tf.subtract(self.reconstruction, input_data), 2.0))
            # axis=1对列求和
            self.errtmp = tf.square(self.reconstruction - input_data)
        self.optimizer = optimizer.minimize(self.cost)

    def calc_total_cost(self, sess):
        return sess.run(self.cost)

    def transform(self, sess):
        return sess.run(self.hidden_out)

    def reconstruct(self, sess):
        return sess.run(self.reconstruction)

# *****************************************************************************************


# *****************************************************************************************
# 20180506
# 1.能够自定义隐藏层层数，及节点数
# 2.输入数据不再是placeholder形式
# 3.会话不在类中定义
# 4.默认指定硬件设备为GPU
# 5.参数初始化方式和v2不一样
class SAEnoplhV6(object):
    def __init__(self,
                 input_data,
                 n_input,
                 stack_size,
                 hidden_size,
                 transfer_function=tf.nn.leaky_relu,
                 optimizer=tf.train.AdamOptimizer()
                 ):
        self.weights = dict()
        self.layer_out = []
        # model
        hidden = input_data
        for i in xrange(stack_size):
            if i == 0:
                input_size = n_input
            else:
                input_size = hidden_size[i-1]

            with tf.variable_scope('hidden_cal%d' % i):
                with tf.device('/cpu:0'):
                    self.weights['w%d' % i] = tf.get_variable(name=('weight%d' % i),
                                                              shape=[input_size, hidden_size[i]],
                                                              dtype=tf.float32)
                    self.weights['b%d' % i] = tf.get_variable(name=('bias%d' % i),
                                                              initializer=tf.zeros([hidden_size[i]], dtype=tf.float32))
                hidden = transfer_function(tf.add(tf.matmul(hidden, self.weights['w%d' % i]), self.weights['b%d' % i]))
                self.layer_out.append(hidden)
                if math.ceil(stack_size/2) == i:
                    self.hidden_out = hidden
                    self.shape = tf.shape(self.hidden_out)

        with tf.variable_scope('output_layer'):
            with tf.device('/cpu:0'):
                self.w_out = tf.get_variable(name="Weight_out",
                                             shape=[hidden_size[stack_size-1], n_input],
                                             dtype=tf.float32)
                self.b_out = tf.get_variable(name="Bias_out",
                                             initializer=tf.zeros([n_input], dtype=tf.float32))

            self.reconstruction = transfer_function(tf.add(tf.matmul(hidden, self.w_out), self.b_out))
        # cost
        with tf.name_scope("cost_errtmp"):
            # 和下面的写法效果是一样的，就是最后一步是求平均
            self.cost = 0.5 * tf.reduce_mean(tf.square(self.reconstruction - input_data), name='cost')
            # 和上面的写法还是一样的，就是最后一步是求和
            # self.cost = 0.5 * tf.reduce_sum(tf.pow(tf.subtract(self.reconstruction, input_data), 2.0))
            # axis=1对列求和
            self.errtmp = tf.square(self.reconstruction - input_data)
        self.optimizer = optimizer.minimize(self.cost)

    def calc_total_cost(self, sess):
        return sess.run(self.cost)

    def transform(self, sess):
        return sess.run(self.hidden_out)

    def reconstruct(self, sess):
        return sess.run(self.reconstruction)

# *****************************************************************************************




# *****************************************************************************************
# 20180415
# 1.能够自定义隐藏层层数，及节点数
# 2.输入数据不再是placeholder形式
# 3.会话不在类中定义
# 4.默认指定硬件设备为GPU
# 5.增加了变学习率的功能
class SAEnoplhV3(object):
    def __init__(self,
                 input_data,
                 n_input,
                 stack_size,
                 hidden_size,
                 transfer_function=tf.nn.softplus,
                 optimizer=tf.train.AdamOptimizer
                 ):
        self.weights = dict()

        lr_init = 0.05        # 初始学习率
        decay_steps = 1000    # 每10步衰减一次
        decay_factor = 0.7  # 学习率衰减率
        # current_step 在学习的过程中要根据训练的步数进行重新赋值，所以设置为variable
        self.current_step = tf.get_variable('current_step', [], initializer=tf.constant_initializer(0), trainable=False)
        self.lr = tf.train.exponential_decay(lr_init, self.current_step, decay_steps, decay_factor, staircase=True)

        # model
        hidden = input_data
        for i in xrange(stack_size):
            if i == 0:
                input_size = n_input
            else:
                input_size = hidden_size[i-1]

            with tf.variable_scope('hidden_cal%d' % i):
                with tf.device('/cpu:0'):
                    self.weights['w%d' % i] = tf.get_variable(name=('weight%d' % i),
                                                              initializer=tf.truncated_normal(
                                                                  [input_size, hidden_size[i]],
                                                                  mean=0.0, stddev=1.0,
                                                                  dtype=tf.float32))
                    self.weights['b%d' % i] = tf.get_variable(name=('bias%d' % i),
                                                              initializer=tf.truncated_normal(
                                                                  [hidden_size[i]],
                                                                  mean=0.0,
                                                                  stddev=1.0,
                                                                  dtype=tf.float32))
                hidden = transfer_function(tf.add(tf.matmul(hidden, self.weights['w%d' % i]), self.weights['b%d' % i]))
                if math.ceil(stack_size/2) == i:
                    self.hidden_out = hidden
                    self.shape = tf.shape(self.hidden_out)

        with tf.variable_scope('output_layer'):
            with tf.device('/cpu:0'):
                self.w_out = tf.get_variable(name="Weight_out",
                                             initializer=tf.truncated_normal([hidden_size[stack_size-1], n_input], mean=0.0, stddev=1.0,
                                                                             dtype=tf.float32))
                self.b_out = tf.get_variable(name="Bias_out",
                                             initializer=tf.truncated_normal([n_input], mean=0.0, stddev=1.0, dtype=tf.float32))

            self.reconstruction = tf.add(tf.matmul(hidden, self.w_out), self.b_out)
        # cost
        with tf.name_scope("cost_errtmp"):
            # 和下面的写法效果是一样的，就是最后一步是求平均
            self.cost = 0.5 * tf.reduce_mean(tf.square(self.reconstruction - input_data), name='cost')
            self.errtmp = tf.square(self.reconstruction - input_data)
        self.optimizer = optimizer(self.lr).minimize(self.cost, global_step=self.current_step)

    def calc_total_cost(self, sess):
        return sess.run(self.cost)

    def transform(self, sess):
        return sess.run(self.hidden_out)

    def reconstruct(self, sess):
        return sess.run(self.reconstruction)

# *****************************************************************************************


# *****************************************************************************************
# 20180505
# 为适应多GPU的设计，设计没有优化器的sae类
class SAEnoplhV5(object):
    def __init__(self,
                 input_data,
                 n_input,
                 stack_size,
                 hidden_size,
                 transfer_function=tf.nn.leaky_relu,
                 ):
        self.weights = dict()
        # model
        hidden = input_data
        for i in xrange(stack_size):
            if i == 0:
                input_size = n_input
            else:
                input_size = hidden_size[i-1]

            with tf.variable_scope('hidden_cal%d' % i):
                with tf.device('/cpu:0'):
                    self.weights['w%d' % i] = tf.get_variable(name=('weight%d' % i),
                                                              shape=[input_size, hidden_size[i]],
                                                              dtype=tf.float32)
                    self.weights['b%d' % i] = tf.get_variable(name=('bias%d' % i),
                                                              initializer=tf.zeros([hidden_size[i]], dtype=tf.float32))
                hidden = transfer_function(tf.add(tf.matmul(hidden, self.weights['w%d' % i]), self.weights['b%d' % i]))
                if math.ceil(stack_size/2) == i:
                    self.hidden_out = hidden
                    self.shape = tf.shape(self.hidden_out)

        with tf.variable_scope('output_layer'):
            with tf.device('/cpu:0'):
                self.w_out = tf.get_variable(name="Weight_out",
                                             shape=[hidden_size[stack_size-1], n_input],
                                             dtype=tf.float32)
                self.b_out = tf.get_variable(name="Bias_out",
                                             initializer=tf.zeros([n_input], dtype=tf.float32))

            self.reconstruction = transfer_function(tf.add(tf.matmul(hidden, self.w_out), self.b_out))
        # cost
            # 和下面的写法效果是一样的，就是最后一步是求平均
        self.cost = 0.5 * tf.reduce_mean(tf.square(self.reconstruction - input_data), name='cost')
        tf.add_to_collection('losses', self.cost)
        self.total_loss = tf.add_n(tf.get_collection('losses'), name='total_loss')
# *****************************************************************************************



# 20171204
# 1.能够自定义隐藏层层数，及节点数
# 2.输入数据不再是placeholder形式
# 3.会话不在类中定义
# 4.修改计算任务硬件分配
# cpu计算：1.0s
# 变量分配到cpu，网络计算子在gpu： 1.45s
class SAEnoplhdevice(object):
    def __init__(self,
                 input_data,
                 n_input,
                 stack_size,
                 hidden_size,
                 transfer_function=tf.nn.softplus,
                 optimizer=tf.train.AdamOptimizer()
                 ):
        self.weights = dict()
        # model
        hidden = input_data
        for i in xrange(stack_size):
            if i == 0:
                input_size = n_input
            else:
                input_size = hidden_size[i-1]
            with tf.device('/cpu:0'):
                with tf.variable_scope('hidden_cal%d' % i):
                    self.weights['w%d' % i] = tf.get_variable(name=('weight%d' % i),
                                                              initializer=tf.truncated_normal(
                                                                  [input_size, hidden_size[i]],
                                                                  mean=0.0, stddev=1.0,
                                                                  dtype=tf.float32))
                    self.weights['b%d' % i] = tf.get_variable(name=('bias%d' % i),
                                                              initializer=tf.truncated_normal(
                                                                  [hidden_size[i]],
                                                                  mean=0.0,
                                                                  stddev=1.0,
                                                                  dtype=tf.float32))
                    hidden = transfer_function(tf.add(tf.matmul(hidden, self.weights['w%d' % i]), self.weights['b%d' % i]))
                    if math.ceil(stack_size/2) == i:
                        self.hidden_out = hidden
                        self.shape = tf.shape(self.hidden_out)

        with tf.device("/cpu:0"):
            with tf.variable_scope('output_layer'):
                self.w_out = tf.get_variable(name="Weight_out",
                                             initializer=tf.truncated_normal([hidden_size[stack_size-1], n_input], mean=0.0, stddev=1.0,
                                                                             dtype=tf.float32))
                self.b_out = tf.get_variable(name="Bias_out",
                                             initializer=tf.truncated_normal([n_input], mean=0.0, stddev=1.0, dtype=tf.float32))

                self.reconstruction = tf.add(tf.matmul(hidden, self.w_out), self.b_out)
            # cost
            with tf.name_scope("cost_errtmp"):
                # self.cost = 0.5 * tf.reduce_mean(tf.square(self.reconstruction - input_data), name='cost')
                self.cost = 0.5 * tf.reduce_sum(tf.pow(tf.subtract(self.reconstruction, input_data), 2.0))
                self.errtmp = tf.reduce_sum(tf.square(self.reconstruction - input_data), axis=1, name='reconstruct_err')
            self.optimizer = optimizer.minimize(self.cost)

    def calc_total_cost(self, sess):
        return sess.run(self.cost)

    def transform(self, sess):
        return sess.run(self.hidden_out)

    def reconstruct(self, sess):
        return sess.run(self.reconstruction)


# 20180322
# 在整理代码的时候发现之前写的预训练过程速度慢，采用速度快的写法写了一个自编码器
class AE_mult_add(object):
    def __init__(self,
                 n_input,
                 hidden_size,
                 transfer_function=tf.nn.leaky_relu,
                 optimizer=tf.train.AdamOptimizer()
                 ):
        # model
        self.input_data = tf.placeholder(tf.float32, [None, n_input])
        with tf.name_scope('encoder'):
            self.w_en = tf.get_variable(name='weight_en',
                                        shape=[n_input, hidden_size],
                                        dtype=tf.float32)
            self.b_en = tf.get_variable(name='bias_en',
                                        initializer=tf.zeros([hidden_size], dtype=tf.float32))
            self.hidden = transfer_function(tf.add(tf.matmul(self.input_data, self.w_en), self.b_en))

        with tf.name_scope('decoder'):
            self.w_de = tf.get_variable(name="weight_de",
                                        shape=[hidden_size, n_input],
                                        dtype=tf.float32)
            self.b_de = tf.get_variable(name="bias_de",
                                        initializer=tf.zeros(n_input, dtype=tf.float32))

            self.reconstruction = transfer_function(tf.add(tf.matmul(self.hidden, self.w_de), self.b_de))
        # cost
        with tf.name_scope("cost_errtmp"):
            # 和下面的写法效果是一样的，就是最后一步是求平均
            self.cost = 0.5 * tf.reduce_mean(tf.square(self.reconstruction - self.input_data), name='cost')
            # 和上面的写法还是一样的，就是最后一步是求和
            # self.cost = 0.5 * tf.reduce_sum(tf.pow(tf.subtract(self.reconstruction, input_data), 2.0))
            # axis=1对列求和
            self.errtmp = tf.reduce_sum(tf.square(self.reconstruction - self.input_data), axis=1, name='reconstruct_err')
        self.optimizer = optimizer.minimize(self.cost)

# class AE_mult_add(object):
#     def __init__(self,
#                  n_input,
#                  hidden_size,
#                  transfer_function=tf.nn.softplus,
#                  optimizer=tf.train.AdamOptimizer()
#                  ):
#         # model
#         self.input_data = tf.placeholder(tf.float32, [None, n_input])
#         with tf.name_scope('encoder'):
#             self.w_en = tf.get_variable(name='weight_en',
#                                         initializer=tf.truncated_normal([n_input, hidden_size],
#                                                                         mean=0.0,
#                                                                         stddev=1.0,
#                                                                         dtype=tf.float32))
#             self.b_en = tf.get_variable(name='bias_en',
#                                         initializer=tf.truncated_normal([hidden_size],
#                                                                         mean=0.0,
#                                                                         stddev=1.0,
#                                                                         dtype=tf.float32))
#             self.hidden = transfer_function(tf.add(tf.matmul(self.input_data, self.w_en), self.b_en))
#
#         with tf.name_scope('decoder'):
#             self.w_de = tf.get_variable(name="weight_de",
#                                         initializer=tf.truncated_normal([hidden_size, n_input],
#                                                                         mean=0.0,
#                                                                         stddev=1.0,
#                                                                         dtype=tf.float32))
#             self.b_de = tf.get_variable(name="bias_de",
#                                         initializer=tf.truncated_normal([n_input],
#                                                                         mean=0.0,
#                                                                         stddev=1.0,
#                                                                         dtype=tf.float32))
#
#             self.reconstruction = tf.add(tf.matmul(self.hidden, self.w_de), self.b_de)
#         # cost
#         with tf.name_scope("cost_errtmp"):
#             # 和下面的写法效果是一样的，就是最后一步是求平均
#             self.cost = 0.5 * tf.reduce_mean(tf.square(self.reconstruction - self.input_data), name='cost')
#             # 和上面的写法还是一样的，就是最后一步是求和
#             # self.cost = 0.5 * tf.reduce_sum(tf.pow(tf.subtract(self.reconstruction, input_data), 2.0))
#             # axis=1对列求和
#             self.errtmp = tf.reduce_sum(tf.square(self.reconstruction - self.input_data), axis=1,
#                                         name='reconstruct_err')
#         self.optimizer = optimizer.minimize(self.cost)


# 20180322
# 使用全连接网络实现AE
#

class AE_denselayer(object):
    def __init__(self,
                 input_data,
                 n_input,
                 hidden_size,
                 transfer_function=tf.nn.softplus,
                 optimizer=tf.train.AdamOptimizer()
                 ):
        # model
        with tf.name_scope('encoder'):
            self.hidden = tf.layers.dense(inputs=input_data,
                                          units=hidden_size,
                                          activation=tf.nn.softplus,
                                          name='encoder')

        with tf.name_scope('decoder'):
            self.reconstruction = tf.layers.dense(inputs=self.hidden,
                                                  units=n_input,
                                                  activation=tf.nn.softplus,
                                                  name='decoder')
        # cost
        with tf.name_scope("cost_errtmp"):
            # 和下面的写法效果是一样的，就是最后一步是求平均
            self.cost = 0.5 * tf.reduce_mean(tf.square(self.reconstruction - input_data), name='cost')
            # 和上面的写法还是一样的，就是最后一步是求和
            # self.cost = 0.5 * tf.reduce_sum(tf.pow(tf.subtract(self.reconstruction, input_data), 2.0))
            # axis=1对列求和
            self.errtmp = tf.reduce_sum(tf.square(self.reconstruction - input_data), axis=1, name='reconstruct_err')
        self.optimizer = optimizer.minimize(self.cost)


# 20180322
# 新版预训练堆叠成堆叠自编码器

class Stack_autoencoder(object):
    def __init__(self,
                 sae_tuple,
                 stack_size,
                 n_input,
                 transfer_function=tf.nn.leaky_relu,
                 optimizer=tf.train.AdamOptimizer()):
        self.n_input = n_input
        self.transfer = transfer_function

        # model
        self.input_data = tf.placeholder(tf.float32, [None, self.n_input], name='SAE_input')
        hidden = self.input_data
        for i in xrange(stack_size):
            with tf.name_scope('hidden_cal%d' % i):
                hidden = self.transfer(tf.add(tf.matmul(hidden, sae_tuple[i].w_en), sae_tuple[i].b_en))
                if math.ceil(stack_size/2) == i:
                    self.hidden_out = hidden

        self.reconstruction = self.transfer(tf.add(tf.matmul(hidden, sae_tuple[stack_size].w_en), sae_tuple[stack_size].b_en))
        # cost
        self.cost = 0.5 * tf.reduce_mean(tf.square(self.reconstruction - self.input_data))
        self.errtmp = tf.reduce_sum(tf.square(self.reconstruction - self.input_data), axis=1, name='reconstruct_err')
        self.optimizer = optimizer.minimize(self.cost)


# 20180322
# 可变隐藏层设计，在之前的基础上加上重建误差和隐藏层特征的padding
# 服务于AD检测器
local_win_width = 2     # 3*2  window is 7*7
local_win_height = 2    # 3*2  window is 7*7
class SAEnoplhV2_with_padding(object):
    def __init__(self,
                 input_data,
                 n_input,
                 stack_size,
                 hidden_size,
                 transfer_function=tf.nn.softplus,
                 optimizer=tf.train.AdamOptimizer()
                 ):
        self.weights = dict()
        # model
        hidden = input_data
        for i in xrange(stack_size):
            if i == 0:
                input_size = n_input
            else:
                input_size = hidden_size[i-1]

            with tf.variable_scope('hidden_cal%d' % i):
                with tf.device('/cpu:0'):
                    self.weights['w%d' % i] = tf.get_variable(name=('weight%d' % i),
                                                              initializer=tf.truncated_normal(
                                                                  [input_size, hidden_size[i]],
                                                                  mean=0.0, stddev=1.0,
                                                                  dtype=tf.float32))
                    self.weights['b%d' % i] = tf.get_variable(name=('bias%d' % i),
                                                              initializer=tf.truncated_normal(
                                                                  [hidden_size[i]],
                                                                  mean=0.0,
                                                                  stddev=1.0,
                                                                  dtype=tf.float32))
                hidden = transfer_function(tf.add(tf.matmul(hidden, self.weights['w%d' % i]), self.weights['b%d' % i]))
                if math.ceil(stack_size/2) == i:
                    self.hidden_out = hidden
                    self.shape = tf.shape(self.hidden_out)

        with tf.variable_scope('output_layer'):
            with tf.device('/cpu:0'):
                self.w_out = tf.get_variable(name="Weight_out",
                                             initializer=tf.truncated_normal([hidden_size[stack_size-1], n_input], mean=0.0, stddev=1.0,
                                                                             dtype=tf.float32))
                self.b_out = tf.get_variable(name="Bias_out",
                                             initializer=tf.truncated_normal([n_input], mean=0.0, stddev=1.0, dtype=tf.float32))

            self.reconstruction = tf.add(tf.matmul(hidden, self.w_out), self.b_out)
        # cost
        with tf.name_scope("cost_errtmp"):
            # 和下面的写法效果是一样的，就是最后一步是求平均
            self.cost = 0.5 * tf.reduce_mean(tf.square(self.reconstruction - input_data), name='cost')
            self.errtmp = tf.reduce_sum(tf.square(self.reconstruction - input_data), axis=1, name='reconstruct_err')
        self.optimizer = optimizer.minimize(self.cost)

        # 对隐藏层输出做padding
        self.shape = tf.shape(self.hidden_out)  # 获得转置后的形状
        self.hidden_pad = tf.pad(tf.reshape(self.hidden_out, [self.shape[1], self.shape[2], self.shape[3]]),
                                 [[local_win_width, local_win_width],
                                  [local_win_height, local_win_height],
                                  [0, 0]],
                                 "SYMMETRIC")  # 对整张图片进行边缘对称pad

        # 对重建误差做padding
        self.errshape = tf.shape(self.errtmp)  # 获得转置后的形状
        self.recon_err = tf.reshape(self.errtmp, [self.errshape[1], self.errshape[2], self.errshape[3]])
        self.recon_err_pad = tf.pad(self.recon_err,
                                    [[local_win_width, local_win_width],
                                     [local_win_height, local_win_height],
                                     [0, 0]],
                                    "SYMMETRIC")  # 对整张图片进行边缘对称pad

        tf.sqrt(self.errshape)