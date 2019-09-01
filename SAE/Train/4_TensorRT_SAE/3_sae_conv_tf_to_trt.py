# -*- coding: utf-8 -*-
# 20180504
# 卷积SAE网络，包括重新训练，模型重载，生成uff文件
# 卷积SAE输入数据方式NHWC
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from sklearn.metrics import roc_curve, auc, roc_auc_score
from read_data import read_txt
import matplotlib.pyplot as plt
import tensorflow as tf
import scipy.io as sio
import numpy as np
import time
import math
import os
import sys
import uff
import tensorrt as trt
from tensorrt.parsers import uffparser

try:
    from PIL import Image
    import pycuda.driver as cuda
    import pycuda.autoinit
    import argparse
except ImportError as err:
    sys.stderr.write("""ERROR: failed to import module ({}) 
Please make sure you have pycuda and the example dependencies installed. 
https://wiki.tiker.net/PyCuda/Installation/Linux
pip(3) install tensorrt[examples]
""".format(err))
    exit(1)

MAX_WORKSPACE = 1 << 30
G_LOGGER = trt.infer.ConsoleLogger(trt.infer.LogSeverity.ERROR)
MAX_BATCHSIZE = 10
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
log_dir = os.path.join(os.getcwd(), '3_sae_conv/')
path = os.path.dirname(os.path.realpath(__file__))

OUTPUT_NAMES = "conv2d_4/LeakyRelu/Maximum"
UFF_OUTPUT_FILENAME = path + "/3_sae_cov.uff"
engine_file = path + "/2_sae_cov.engine"


# 1 读取高光谱图像数据
# 测试用的plane数据
load_plane = os.path.join(os.getcwd(), '../hyper_data/plane.mat')
load_plane_label = os.path.join(os.getcwd(), '../hyper_data/plane_label.mat')

plane = sio.loadmat(load_plane)
label = sio.loadmat(load_plane_label)

plane_image = plane['array']
trX = plane_image.reshape(-1, 100, 100, 126)
load_label = label['array'].reshape(10000)/255

kk_size = 1


class Modelconfig(object):
    n_samples = trX.shape[1] * trX.shape[2]                       # 样本的个数
    input_size = trX.shape[3]                                     # 输入层节点数
    training_epochs = 2000                                 # 训练的轮数
    batch_size = 10000                                              # 更新一次参数训练的样本数
    display_step = 1                                              # 打印信息的步数
    stack_size = 3                                                # 自编码器的隐藏层层数
    hidden_size = [166, 20, 166, 83, 166]                                  # 各个隐藏层的节点数
    kernel_size = [kk_size, kk_size, kk_size, kk_size, kk_size, kk_size, kk_size]
    output_kernel_size = kk_size
    n_hidden_size = hidden_size[int(math.floor(stack_size / 2))]  # coding层的输出维度
    pretrain_lr = [0.005, 0.005, 0.005, 0.005, 0.005]             # 预训练不同隐藏层之间的学习率
    finetune_lr = 0.005                                           # 微调训练的学习率
    is_pretrain = 1                                               # 是否预训练开关
    is_retrain = input("please input mode, 1 to retrain, 0 to restore ")
    transfer_function = tf.nn.softplus                            # 激活函数
    lr = 0.00005
    use_gpu = 1


class CNNetwork(object):
    def __init__(self, inputs, stack_size, kernel_size, hidden_size, using_gpu):
        """Takes the MNIST inputs and mode and outputs a tensor of logits."""
        # Input Layer
        # Reshape X to 4-D tensor: [batch_size, width, height, channels]
        # MNIST images are 28x28 pixels, and have one color channel
        inputs = tf.reshape(inputs, [-1, inputs.shape[0], inputs.shape[1], inputs.shape[2]])

        # When running on GPU, transpose the data from channels_last (NHWC) to
        # channels_first (NCHW) to improve performance.
        # See https://www.tensorflow.org/performance/performance_guide#data_formats
        if using_gpu:
            device_name = "gpu:0"
            data_format = 'channels_first'
            inputs = tf.transpose(inputs, [0, 3, 1, 2])
        else:
            device_name = "cpu:0"
            data_format = 'channels_last'

        with tf.device(device_name):
            hidden = inputs
            for i in xrange(stack_size):
                hidden = tf.layers.conv2d(
                    inputs=hidden,
                    filters=hidden_size[i],
                    kernel_size=[kernel_size[i], kernel_size[i]],
                    padding='same',
                    activation=tf.nn.leaky_relu,
                    data_format=data_format)

                if math.floor(stack_size / 2) == i:
                    # 特征层处理：1.获得特征并转置；2.转置后padding
                    self.hidden_out = hidden

            self.conv_out = tf.layers.conv2d(
                inputs=hidden,
                filters=config.input_size,
                kernel_size=[config.output_kernel_size, config.output_kernel_size],
                padding='same',
                activation=tf.nn.leaky_relu,
                data_format=data_format)

            self.cost = 0.5 * tf.reduce_mean(tf.pow(tf.subtract(self.conv_out, inputs), 2.0))
            # 重建误差处理：1.重建误差计算；2.重建误差转置；3重建误差reshape；4.重建误差padding
            self.recon_err_tmp = tf.square(self.conv_out - inputs)
            recon_err_tmp = tf.transpose(self.recon_err_tmp, [0, 2, 3, 1])  # 将中间隐藏层结果转置，让通道数在最后一维
            self.errshape = tf.shape(recon_err_tmp)  # 获得转置后的形状
            self.recon_err = tf.reduce_sum(tf.reshape(recon_err_tmp, [-1, self.errshape[3]]), axis=1)

            optimizer = tf.train.AdamOptimizer(learning_rate=config.lr)
            self.optimizer = optimizer.minimize(self.cost)


class CNNetworkv2(object):
    def __init__(self, inputs, stack_size, kernel_size, hidden_size, using_gpu):
        """Takes the MNIST inputs and mode and outputs a tensor of logits."""

        # inputs = tf.reshape(inputs, (-1, inputs.shape[0], inputs.shape[1], inputs.shape[2]))
        if using_gpu:
            device_name = "gpu:0"
            # data_format = 'channels_first'
            # inputs = tf.transpose(inputs, [0, 3, 1, 2])
        else:
            device_name = "cpu:0"
            # data_format = 'channels_last'

        with tf.device(device_name):
            hidden = inputs
            for i in xrange(stack_size):
                hidden = tf.layers.conv2d(
                    inputs=hidden,
                    filters=hidden_size[i],
                    kernel_size=[kernel_size[i], kernel_size[i]],
                    padding='same',
                    activation=tf.nn.leaky_relu)

                if math.floor(stack_size / 2) == i:
                    # 特征层处理：1.获得特征并转置；2.转置后padding
                    self.hidden_out = hidden

            self.conv_out = tf.layers.conv2d(
                inputs=hidden,
                filters=config.input_size,
                kernel_size=[config.output_kernel_size, config.output_kernel_size],
                padding='same',
                activation=tf.nn.leaky_relu)

            self.cost = 0.5 * tf.reduce_mean(tf.pow(tf.subtract(self.conv_out, inputs), 2.0))
            # 重建误差处理：1.重建误差计算；2.重建误差转置；3重建误差reshape；4.重建误差padding
            self.recon_err = tf.reduce_sum(tf.square(self.conv_out - inputs), axis=3)

            optimizer = tf.train.AdamOptimizer(learning_rate=config.lr)
            self.optimizer = optimizer.minimize(self.cost)


# API CHANGE: Try to generalize into a utils function
# Run inference on device
def infer(context, input_img, batch_size):
    # load engine
    engine = context.get_engine()
    assert(engine.get_nb_bindings() == 2)
    # create output array to receive data
    dims = engine.get_binding_dimensions(1).to_DimsCHW()
    elt_count = dims.C() * dims.H() * dims.W() * batch_size
    # convert input data to Float32
    input_img = input_img.astype(np.float32)
    # Allocate pagelocked memory
    output = cuda.pagelocked_empty(elt_count, dtype=np.float32)

    a = input_img.size
    b = input_img.dtype.itemsize

    # alocate device memory
    d_input = cuda.mem_alloc(input_img.size * input_img.dtype.itemsize)
    d_output = cuda.mem_alloc(output.size * output.dtype.itemsize)

    bindings = [int(d_input), int(d_output)]

    stream = cuda.Stream()

    # transfer input data to device
    cuda.memcpy_htod_async(d_input, input_img, stream)
    # execute model
    context.enqueue(batch_size, bindings, stream.handle, None)
    # transfer predictions back
    cuda.memcpy_dtoh_async(output, d_output, stream)

    # return predictions
    return output


# 建立sae图
config = Modelconfig()
with tf.Graph().as_default():
    image = tf.placeholder(tf.float32, shape=(None, trX.shape[1], trX.shape[2], trX.shape[3]))
    cnn_model = CNNetworkv2(inputs=image,
                            stack_size=config.stack_size,
                            kernel_size=config.kernel_size,
                            hidden_size=config.hidden_size,
                            using_gpu=config.use_gpu)

    init = tf.global_variables_initializer()
    saver = tf.train.Saver(tf.global_variables())
    checkpoint_path = os.path.join(log_dir, 'model.ckpt')
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.5)

    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:

        # 如果不进行重新训练，则进入模型重载模式
        if config.is_retrain == 0:
            saver.restore(sess, checkpoint_path)
            print("Model restored")
        # 否则进行模型重新训练
        else:
            sess.run(init)
            variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
            a = sess.run(variables)
            # summary_writer = tf.summary.FileWriter(log_dir, sess.graph)  # 创建图写入器并写文件
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(coord=coord)
            print("Finetune starting!")

            for epoch in range(config.training_epochs):
                start_time = time.time()
                avg_cost = 0.
                total_batch = int(config.n_samples / config.batch_size)
                for _ in range(total_batch):
                    cost, _ = sess.run((cnn_model.cost, cnn_model.optimizer), feed_dict={image: trX})
                    avg_cost += cost / config.n_samples * config.batch_size
                # Display logs per epoch step
                if epoch % config.display_step == 0:
                    print("Epoch:", '%d,' % (epoch + 1), "Cost:", "{:.9f}".format(avg_cost),
                          "Time/Epoch is ", (time.time() - start_time))

            # recon_err = sess.run(cnn_model.recon_err, feed_dict={image: trX})
            # recon_err = np.reshape(recon_err, [10000])
            # roc_auc1 = roc_auc_score(load_label, recon_err)
            # print("AUC is ", roc_auc1)
            while True:
                confirm_key = raw_input("Do you want to save model? y/n: ")
                if confirm_key == 'y':
                    print("Starting save model...")
                    saver.save(sess, checkpoint_path)
                    print("Save successfully!")
                    break
                elif confirm_key == 'n':
                    break
            coord.request_stop()
            coord.join(threads)

        time1 = time.time()
        output = sess.run(cnn_model.conv_out, feed_dict={image: trX})
        output = np.reshape(output, [100, 100, 126])
        print ("time is ", time.time()-time1)

        # 计算重建误差进而计算AUC
        graphdef = tf.get_default_graph().as_graph_def()
        frozen_graph = tf.graph_util.convert_variables_to_constants(sess, graphdef, output_node_names=[OUTPUT_NAMES])
        tf_model = tf.graph_util.remove_training_nodes(frozen_graph)


# cpp_out = read_txt("/home/ubuntu-server/my_file/master_project/TensorRT-3.0.4/data/sae/sae_conv_out.txt")
# cpp_out = np.reshape(cpp_out, (100, 100, 126))
# plt.imshow(100*cpp_out[:, :, 10])
# plt.show()
# 如果要生成uff文件，则取消下面这个函数的屏蔽, 有时候会出错，考虑一下OUTPUT_NAMES是否需要加中括号
uff.from_tensorflow(graphdef=frozen_graph,
                    output_filename=UFF_OUTPUT_FILENAME,
                    output_nodes=[OUTPUT_NAMES],
                    text=True)


# uff_model = uff.from_tensorflow(tf_model, [OUTPUT_NAMES])
# parser = uffparser.create_uff_parser()
# parser.register_input("Placeholder", (126, 100, 100), 1)
# parser.register_output(OUTPUT_NAMES)
# # parser.register_output("hidden_cal1/Relu")
#
# engine = trt.utils.uff_to_trt_engine(G_LOGGER,
#                                      uff_model,
#                                      parser,
#                                      MAX_BATCHSIZE,
#                                      MAX_WORKSPACE)
# assert engine
# # parser.destroy()
# context = engine.create_execution_context()
# print("\n| TEST CASE | PREDICTION |")
# time2 = time.time()
# out = infer(context, plane_image, 1)
#
# print("time2 is ", time.time()-time2)
# out = np.reshape(out, (126, 100, 100))
# # err = out - output
# # err = np.sum(np.square(err))
# # err = np.sum(np.square(err), axis=1)
# #
# # print (err)
# plt.imshow(100*output[:, :, 10])
# plt.show()
# #
# plt.imshow(100*out[10, :, :])
# plt.show()

