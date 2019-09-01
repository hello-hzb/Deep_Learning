# -*- coding: utf-8 -*-
# 20180505
# 好的检测精度能达到0.87
# 双GPU卡的加速训练
# 使用placeholder的形式设计
#          batch_size=100                  batch_size=1000     batch_size=5000
# CPU       89.8695218563  81.8760778904    21.338039875       12.8481109142
# 1GPU      49.5230431557  47.2495799065    6.92366290092      3.24300599098
# 2GPU      35.4396328926  30.9641129971    5.36277699471      3.28717112541

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from Autoencoder import SAEnoplhV5
from data_preprocess import get_random_block_from_data, OneWindow_Edge_Sel
from sklearn.metrics import roc_curve, auc, roc_auc_score
import matplotlib.pyplot as plt
import tensorflow as tf
import scipy.io as sio
import re
import numpy as np
import time
import math
import os
os.environ["CUDA_VISIBLE_DEVICES"] = " "
log_dir = '0_sae_multi_GPU_placeholder/'


# 1 读取高光谱图像数据
# 测试用的plane数据
load_plane = os.path.join(os.getcwd(), '../hyper_data/plane.mat')
load_plane_label = os.path.join(os.getcwd(), '../hyper_data/plane_label.mat')

plane = sio.loadmat(load_plane)
label = sio.loadmat(load_plane_label)

plane_image = plane['array']
trX = plane_image.reshape(100, 100, 126)
load_label = label['array'].reshape(10000)/255

local_win_width = 2     # 3*2  window is 7*7
local_win_height = 2    # 3*2  window is 7*7


class Modelconfig(object):
    n_samples = trX.shape[0] * trX.shape[1]                       # 样本的个数
    input_size = trX.shape[2]                                     # 输入层节点数
    training_epochs = 200                                         # 训练的轮数
    batch_size = 5000                                               # 更新一次参数训练的样本数
    display_step = 1                                              # 打印信息的步数
    stack_size = 3                                                # 自编码器的隐藏层层数
    hidden_size = [166, 20, 166]                                  # 各个隐藏层的节点数
    n_hidden_size = hidden_size[int(math.floor(stack_size / 2))]  # coding层的输出维度
    pretrain_lr = [0.005, 0.005, 0.005, 0.005, 0.005]             # 预训练不同隐藏层之间的学习率
    finetune_lr = 0.005                                         # 微调训练的学习率
    is_pretrain = 1                                               # 是否预训练开关
    is_retrain = input("please input mode, 1 to retrain, 0 to restore ")
    transfer_function = tf.nn.softplus                            # 激活函数
    num_gpus = 1
    num_epochs_per_decay = 200
    initial_lr = 0.00005
    lr_dacay_factor = 0.1


config = Modelconfig()
# image = tf.placeholder(tf.float32, [None, config.input_size])
# sae = SAEModel(input_data=image,
#                n_input=config.input_size,
#                stack_size=config.stack_size,
#                hidden_size=config.hidden_size,
#                optimizer=tf.train.AdamOptimizer(learning_rate=0.00005))


def tower_loss(scope, images):
    # # Build inference Graph.
    sae = SAEnoplhV5(input_data=images,
                     n_input=config.input_size,
                     stack_size=config.stack_size,
                     hidden_size=config.hidden_size)  # 将权重和偏置放到定义到CPU中，性能有10%的提升

    # Assemble all of the losses for the current tower only.
    losses = tf.get_collection('losses', scope)

    # Calculate the total loss for the current tower.
    total_loss = tf.add_n(losses, name='total_loss')

    # Attach a scalar summary to all individual losses and the total loss; do the
    # same for the averaged version of the losses.
    for l in losses + [total_loss]:
        # Remove 'tower_[0-9]/' from the name in case this is a multi-GPU training
        # session. This helps the clarity of presentation on tensorboard.
        loss_name = re.sub('%s_[0-9]*/' % "Tower", '', l.op.name)
        tf.summary.scalar(loss_name, l)
    return total_loss


def average_gradients(tower_grads):
    """Calculate the average gradient for each shared variable across all towers.

    Note that this function provides a synchronization point across all towers.

    Args:
      tower_grads: List of lists of (gradient, variable) tuples. The outer list
        is over individual gradients. The inner list is over the gradient
        calculation for each tower.
    Returns:
       List of pairs of (gradient, variable) where the gradient has been averaged
       across all towers.
    """
    average_grads = []
    for grad_and_vars in zip(*tower_grads):
        # Note that each grad_and_vars looks like the following:
        #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
        grads = []
        for g, _ in grad_and_vars:
            # Add 0 dimension to the gradients to represent the tower.
            expanded_g = tf.expand_dims(g, 0)

            # Append on a 'tower' dimension which we will average over below.
            grads.append(expanded_g)

        # Average over the 'tower' dimension.
        grad = tf.concat(axis=0, values=grads)
        grad = tf.reduce_mean(grad, 0)

        # Keep in mind that the Variables are redundant because they are shared
        # across towers. So .. we will just return the first tower's pointer to
        # the Variable.
        v = grad_and_vars[0][1]
        grad_and_var = (grad, v)
        average_grads.append(grad_and_var)
    return average_grads

def train():
    """Train CIFAR-10 for a number of steps."""
    with tf.Graph().as_default(), tf.device('/cpu:0'):
        # Create a variable to count the number of train() calls. This equals the
        # number of batches processed * FLAGS.num_gpus.
        global_step = tf.get_variable(
            'global_step', [],
            initializer=tf.constant_initializer(0), trainable=False)

        # Calculate the learning rate schedule.
        num_batches_per_epoch = (config.n_samples / config.batch_size)  # num of updating times per epoch
        decay_steps = int(num_batches_per_epoch * config.num_epochs_per_decay)  # updating every step

        # Decay the learning rate exponentially based on the number of steps.
        lr = tf.train.exponential_decay(config.initial_lr,
                                        global_step,
                                        decay_steps,
                                        config.lr_dacay_factor,
                                        staircase=True)
        opt = tf.train.AdamOptimizer(0.00005)
        # dataset = tf.data.Dataset.from_tensor_slices(plane_image)
        # dataset = dataset.repeat(config.training_epochs*config.num_gpus)  #
        # dataset = dataset.batch(config.batch_size)
        # iterator = dataset.make_initializable_iterator()
        # images = iterator.get_next()

        # input_image = tf.constant(plane_image)
        input1 = tf.placeholder(tf.float32, [None, config.input_size])
        input2 = tf.placeholder(tf.float32, [None, config.input_size])
        input_image = [input1, input2]

        # images = tf.placeholder(tf.float32, [None, config.input_size])
        # batch_queue = tf.contrib.slim.prefetch_queue.prefetch_queue(
        #       [images, labels], capacity=2 * FLAGS.num_gpus)
        # Calculate the gradients for each model tower.
        tower_grads = []
        with tf.variable_scope(tf.get_variable_scope()):
            for i in xrange(config.num_gpus):
                with tf.device('/gpu:%d' % i):
                    with tf.name_scope('%s_%d' % ("SAE_Tower", i)) as scope:
                        # Dequeues one batch for the GPU
                        # image_batch, label_batch = batch_queue.dequeue()
                        # Calculate the loss for one tower of the CIFAR model. This function
                        # constructs the entire CIFAR model but shares the variables across
                        # all towers.
                        # images = tf.train.shuffle_batch([input_image],
                        #                                 batch_size=config.batch_size,
                        #                                 capacity=10000,
                        #                                 num_threads=10,
                        #                                 min_after_dequeue=100,
                        #                                 enqueue_many=True)
                        # loss = tower_loss(scope, images)
                        loss = tower_loss(scope, input_image[i])

                        # Reuse variables for the next tower.
                        tf.get_variable_scope().reuse_variables()

                        # Retain the summaries from the final tower.
                        summaries = tf.get_collection(tf.GraphKeys.SUMMARIES, scope)

                        # Calculate the gradients for the batch of data on this CIFAR tower.
                        grads = opt.compute_gradients(loss)

                        # Keep track of the gradients across all towers.
                        tower_grads.append(grads)

        # We must calculate the mean of each gradient. Note that this is the
        # synchronization point across all towers.
        grads = average_gradients(tower_grads)

        # Add a summary to track the learning rate.
        summaries.append(tf.summary.scalar('learning_rate', lr))

        # Add histograms for gradients.
        for grad, var in grads:
            if grad is not None:
                summaries.append(tf.summary.histogram(var.op.name + '/gradients', grad))

        # Apply the gradients to adjust the shared variables.
        apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)

        # Add histograms for trainable variables.
        for var in tf.trainable_variables():
            summaries.append(tf.summary.histogram(var.op.name, var))

        # Track the moving averages of all trainable variables.
        variable_averages = tf.train.ExponentialMovingAverage(0.9999, global_step)
        variables_averages_op = variable_averages.apply(tf.trainable_variables())

        # Group all updates to into a single train op.
        train_op = tf.group(apply_gradient_op, variables_averages_op)

        # Create a saver.
        # saver = tf.train.Saver(tf.global_variables())

        # Build the summary operation from the last tower summaries.
        # summary_op = tf.summary.merge(summaries)

        # Build an initialization operation to run below.
        # init = tf.global_variables_initializer()
        init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())

        # Start running operations on the Graph. allow_soft_placement must be set to
        # True to build towers on GPU, as some of the ops do not have GPU
        # implementations.
        sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=True))
        sess.run(init)
        # sess.run(iterator.initializer)

        # Start the queue runners.
        tf.train.start_queue_runners(sess=sess)
        start_time = time.time()
        # summary_writer = tf.summary.FileWriter(log_dir, sess.graph)
        for epoch in range(config.training_epochs):

            avg_cost = 0.
            total_batch = int(config.n_samples / (config.num_gpus*config.batch_size))
            for _ in range(total_batch):
                batch_0 = get_random_block_from_data(plane_image, config.batch_size)
                batch_1 = get_random_block_from_data(plane_image, config.batch_size)

                # cost, _ = sess.run([loss, train_op], feed_dict={images: batch_xs})
                cost, _ = sess.run([loss, train_op], feed_dict={input_image[0]: batch_0,
                                                                input_image[1]: batch_1})

                # cost = sess.run(sae.cost, feed_dict={image: batch_xs})

                avg_cost += cost / config.n_samples * config.batch_size
            # 生成timeline文件已经在tensorboard中加入内存和运行时间的记录，但是会导致运行时间增加
            # summary_writer.add_run_metadata(run_metadata, 'step%03d' % epoch)
            # fetched_timeline = timeline.Timeline(run_metadata.step_stats)
            # chrome_trace = fetched_timeline.generate_chrome_trace_format()
            # with open('timeline_gpu.json', 'w') as f:
            #     f.write(chrome_trace)

            # Display logs per epoch step
            # if epoch % config.display_step == 0:
            #     print("Epoch:", '%d,' % (epoch + 1), "Cost:", "{:.9f}".format(avg_cost),
            #           "Time/Epoch is ", (time.time() - start_time))
        print("TOTAL TIME IS ", time.time()-start_time)
        # # for step in xrange(FLAGS.max_steps):
        # for step in xrange(100000):
        #     batch_xs = get_random_block_from_data(plane_image, config.batch_size)
        #     start_time = time.time()
        #     _, loss_value = sess.run([train_op, loss], feed_dict={images: batch_xs})
        #     duration = time.time() - start_time
        #
        #     assert not np.isnan(loss_value), 'Model diverged with loss = NaN'
        #
        #     if step % 10 == 0:
        #         num_examples_per_step = config.batch_size * config.num_gpus
        #         examples_per_sec = num_examples_per_step / duration
        #         sec_per_batch = duration / config.num_gpus
        #
        #     format_str = ('%s: step %d, loss = %.2f (%.1f examples/sec; %.3f '
        #                   'sec/batch)')
        #     print (format_str % (datetime.now(), step, loss_value,
        #                          examples_per_sec, sec_per_batch))
        #
        #     if step % 100 == 0:
        #         summary_str = sess.run(summary_op, feed_dict={images: batch_xs})
        #         summary_writer.add_summary(summary_str, step)
        #
        #     # Save the model checkpoint periodically.
        #     if step % 1000 == 0 or (step + 1) == 10000:
        #         checkpoint_path = os.path.join(log_dir, 'model.ckpt')
        #         saver.save(sess, checkpoint_path, global_step=step)


train()
