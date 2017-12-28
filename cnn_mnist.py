# -*- coding: utf-8 -*-
"""
Created on Thu Sep 21 19:03:02 2017

@author: volde
"""
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot = True)
import tensorflow as tf

import time

#==============================================================================
# import numpy as np
#==============================================================================

start = time.time()

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)
    
def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)
    
def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides = [1, 1, 1, 1], padding = 'SAME')
    
def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize = [1, 2, 2, 1], strides= [1, 2, 2, 1],
                          padding = 'SAME')

'''我们通过为输入图像和目标输出类别创建节点，来开始构建计算图。'''
x = tf.placeholder('float', shape = [None, 784])
y_ = tf.placeholder('float', shape = [None, 10])
    
W_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])
x_image = tf.reshape(x, [-1, 28, 28, 1])
#上面的x进行reshape后的第一维是-1，应该是指不限制这一维度，这一维度应该是指输入图片的数量。

'''我们把x_image和权值向量进行卷积，加上偏置项，然后应用ReLU激活函数，
最后进行max pooling。'''
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)     #这两货开头的h是指hidden

'''为了构建一个更深的网络，我们会把几个类似的层堆叠起来。第二层中，每个5x5的patch
会得到64个特征（第一层卷积层输入1个特征，输出32个特征；第二层卷积层输入32个特征，
输出64个特征）'''
W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])

#注意，这里不需要再定义x_image了，因为第一层卷积层的输出h_pool1就是这一层的输入。
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

'''密集连接层：
现在，图片尺寸减小到7x7，我们加入一个有1024个神经元的全连接层，用于处理整个图片。
我们把池化层输出的张量reshape成一些向量，乘上权重矩阵，加上偏置，然后对其使用ReLU'''
W_fc1 = weight_variable([7 * 7 * 64, 1024]) #有意思，前三个数字乘起来了？
b_fc1 = bias_variable([1024])

h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

'''为了减少过拟合，我们在输出层之前加入dropout。我们用一个placeholder来代表一个神经元
的输出在dropout中保持不变的概率。这样我们可以在训练过程中启用dropout，在测试过程中关闭
dropout。 TensorFlow的tf.nn.dropout操作除了可以屏蔽神经元的输出外，还会自动处理神经
元输出值的scale。所以用dropout的时候可以不用考虑scale。'''
keep_prob = tf.placeholder('float')
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

'''输出层：最后，我们添加一个softmax层，就像前面的单层softmax regression一样。'''
W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])

y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2), b_fc2)

'''为了进行训练和评估，我们使用与之前简单的单层SoftMax神经网络模型几乎相同的一套代码，
只是我们会用更加复杂的ADAM优化器来做梯度最速下降，在feed_dict中加入额外的参数
keep_prob来控制dropout比例。然后每100次迭代输出一次日志。'''

cross_entropy = -tf.reduce_sum(y_*tf.log(y_conv))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, 'float'))
sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())
for i in range(1200):
    batch = mnist.train.next_batch(50)
    if i%100 == 0:
        train_accuracy = accuracy.eval(feed_dict={
                            x:batch[0], y_: batch[1], keep_prob: 1.0})
        print("step %d, training accuracy %g"%(i, train_accuracy))
    train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})

print("test accuracy %g"%accuracy.eval(feed_dict={
    x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))

end = time.time()
total = end - start
print("\t\t total time is: ")
print(total)




