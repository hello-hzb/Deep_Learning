# -*- coding: utf-8 -*-
# 20180502
# sae网络包括模型重新训练，模型重载，生成uff文件
# 重建误差求平方的操作tensorRT不支持，所以只输出重建图像和输入的差
# 使用模型重载后启动session获取结果(0.4108s)相比训练后获得结果(0.0145)的耗时更长，可作为大论文的一个点
# 网络的输出，tensorflow和tensorRT的输出做差求平方后求和的结果1.537827e-09，很小

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from Autoencoder import SAEnoplhV2, SAEnoplhV4
from data_preprocess import get_random_block_from_data, OneWindow_Edge_Sel
from read_data import read_txt
from sklearn.metrics import roc_curve, auc, roc_auc_score
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
G_LOGGER = trt.infer.ConsoleLogger(trt.infer.LogSeverity.INFO)
MAX_BATCHSIZE = 20000
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
log_dir = os.path.join(os.getcwd(), '1_sae_tf_to_trt/')
path = os.path.dirname(os.path.realpath(__file__))

OUTPUT_NAMES = "output_layer/LeakyRelu/Maximum"
# OUTPUT_NAMES = "cost_errtmp/sub"
UFF_OUTPUT_FILENAME = path + "/1_sae.uff"

# 1 读取高光谱图像数据
# 测试用的plane数据
load_plane = os.path.join(os.getcwd(), '../hyper_data/plane.mat')
load_plane_label = os.path.join(os.getcwd(), '../hyper_data/plane_label.mat')

plane = sio.loadmat(load_plane)
label = sio.loadmat(load_plane_label)

plane_image = plane['array']
trX = plane_image.reshape(100, 100, 126)
load_label = label['array'].reshape(10000)/255


class Modelconfig(object):
    n_samples = trX.shape[0] * trX.shape[1]                       # 样本的个数
    input_size = trX.shape[2]                                     # 输入层节点数
    training_epochs = 2000                                         # 训练的轮数
    batch_size = 10000                                               # 更新一次参数训练的样本数
    display_step = 1                                              # 打印信息的步数
    stack_size = 3                                                # 自编码器的隐藏层层数
    hidden_size = [166, 20, 166]                                  # 各个隐藏层的节点数
    n_hidden_size = hidden_size[int(math.floor(stack_size / 2))]  # coding层的输出维度
    pretrain_lr = [0.005, 0.005, 0.005, 0.005, 0.005]             # 预训练不同隐藏层之间的学习率
    finetune_lr = 0.0005                                         # 微调训练的学习率
    is_pretrain = 1                                               # 是否预训练开关
    is_retrain = input("please input mode, 1 to retrain, 0 to restore ")
    transfer_function = tf.nn.softplus                            # 激活函数


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
    image = tf.placeholder(tf.float32, [None, config.input_size])
    sae = SAEnoplhV4(input_data=image,
                     n_input=config.input_size,
                     stack_size=config.stack_size,
                     hidden_size=config.hidden_size,
                     optimizer=tf.train.AdamOptimizer(learning_rate=0.00005)
                     )  # 将权重和偏置放到定义到CPU中，性能有10%的提升

    init = tf.global_variables_initializer()
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.5)  # 这个必须加，否者会出错
    saver = tf.train.Saver(tf.global_variables())
    checkpoint_path = os.path.join(log_dir, 'model.ckpt')

    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
        # 如果不进行重新训练，则进入模型重载模式
        if config.is_retrain == 0:
            saver.restore(sess, checkpoint_path)
            print("Model restored")
        # 否则进行模型重新训练
        else:
            sess.run(init)
            # variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
            # a = sess.run(variables)
            # summary_writer = tf.summary.FileWriter(log_dir, sess.graph)  # 创建图写入器并写文件
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(coord=coord)
            print("Finetune starting!")
            for epoch in range(config.training_epochs):
                # sae.current_step = epoch
                start_time = time.time()
                avg_cost = 0.
                total_batch = int(config.n_samples / config.batch_size)
                for _ in range(total_batch):
                    # batch_xs = get_random_block_from_data(plane_image, config.batch_size)
                    cost, _, = sess.run((sae.cost, sae.optimizer), feed_dict={image: plane_image})
                    avg_cost += cost / config.n_samples * config.batch_size
                # Display logs per epoch step
                if epoch % config.display_step == 0:
                    print("Epoch:", '%d,' % (epoch + 1), "Cost:", "{:.9f}".format(avg_cost),
                          "Time/Epoch is ", (time.time() - start_time))
            # time1 = time.time()
            # output = sess.run(sae.reconstruction, feed_dict={image: plane_image})
            # print("time is ", time.time() - time1)
            # 使用Tensorflow的输出计算AUC, 如果结果较好，点击y保存
            recon_err = sess.run(sae.errtmp, feed_dict={image: plane_image})
            recon_err = np.sum(recon_err, axis=1)
            roc_auc1 = roc_auc_score(load_label, recon_err)
            print("AUC is ", roc_auc1)
            while True:
                confirm_key = raw_input("Do you want to save model? y/n: ")
                if confirm_key == 'y':
                    print("Starting save model...")
                    saver.save(sess, checkpoint_path)
                    print("Save successfully!")
                    break
                elif confirm_key == 'n':
                    break
            saver.save(sess, checkpoint_path)
            coord.request_stop()
            coord.join(threads)

        # output = sess.run(sae.reconstruction, feed_dict={image: plane_image})
        tf_inference_time = time.time()
        output = sess.run(sae.reconstruction, feed_dict={image: plane_image})
        print("Tensorflow inference time is ", time.time() - tf_inference_time)

        graphdef = tf.get_default_graph().as_graph_def()
        frozen_graph = tf.graph_util.convert_variables_to_constants(sess, graphdef, output_node_names=[OUTPUT_NAMES])
        tf_model = tf.graph_util.remove_training_nodes(frozen_graph)

# # 对比cpptensorRT省生成的结果
# cpp_out = read_txt('/home/ubuntu-server/my_file/master_project/SAE_AD_20180427/4_TensorRT_sae/sae_out.txt')
# err = np.sum(np.square(output-cpp_out))
# print(err)
# # 如果要生成uff文件，则取消下面这个函数的屏蔽, 有时候会出错，考虑一下OUTPUT_NAMES是否需要加中括号
# uff.from_tensorflow(graphdef=frozen_graph,
#                     output_filename=UFF_OUTPUT_FILENAME,
#                     output_nodes=[OUTPUT_NAMES],
#                     text=True)
# #
uff_model = uff.from_tensorflow(tf_model, [OUTPUT_NAMES])
parser = uffparser.create_uff_parser()
parser.register_input("Placeholder", (126, 1, 1), 1)
parser.register_output(OUTPUT_NAMES)

engine = trt.utils.uff_to_trt_engine(G_LOGGER,
                                     uff_model,
                                     parser,
                                     MAX_BATCHSIZE,
                                     MAX_WORKSPACE)
assert engine
# parser.destroy()
# runtime = trt.infer.create_infer_runtime(G_LOGGER)
context = engine.create_execution_context()

print("\n| TEST CASE | PREDICTION |")
trt_inference_time = time.time()
out_tmp = []
for i in xrange(10000):
    out = infer(context, plane_image[i, :], 1)
    out_tmp.append(out)
print("trt inference time is", time.time()-trt_inference_time)
out = np.array(out_tmp)

# err = np.sum(np.square(out), axis=1)
# roc_auc1 = roc_auc_score(load_label, err)
# print("AUC is ", roc_auc1)
# print (err)
err = np.sum(np.square(out-output))
print (err)
