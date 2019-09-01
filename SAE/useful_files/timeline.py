# -*- coding: utf-8 -*-
# 20171115
# 跟踪计算过程并测试运行时间
# 生成的.json文件需要通过google浏览器打开, chrome://tracing/

import tensorflow as tf
from tensorflow.python.client import timeline
import os
log_dir = os.path.join(os.getcwd(), 'log_dir_AD/')

a = tf.random_normal([2000, 5000])
b = tf.random_normal([5000, 1000])
res = tf.matmul(a, b)

with tf.Session() as sess:
    # add additional options to trace the session execution
    summary_writer = tf.summary.FileWriter(log_dir, sess.graph)  # 创建图写入器并写文件

    options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
    run_metadata = tf.RunMetadata()
    sess.run(res, options=options, run_metadata=run_metadata)

    # Create the Timeline object, and write it to a json file
    # summary_writer.add_run_metadata(run_metadata, 'step%03d' % epoch)
    summary_writer.add_run_metadata(run_metadata, 'step')

    fetched_timeline = timeline.Timeline(run_metadata.step_stats)
    chrome_trace = fetched_timeline.generate_chrome_trace_format()
    with open('timeline_01.json', 'w') as f:
        f.write(chrome_trace)











'''with tf.Session() as sess:
    # add additional options to trace the session execution
    options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
    run_metadata = tf.RunMetadata()
    for _ in xrange(2):
        sess.run(res, options=options, run_metadata=run_metadata)
        print "hello!"

    # Create the Timeline object, and write it to a json file
    fetched_timeline = timeline.Timeline(run_metadata.step_stats)
    chrome_trace = fetched_timeline.generate_chrome_trace_format()
    with open('timeline_01.json', 'w') as f:
        f.write(chrome_trace)


'''
