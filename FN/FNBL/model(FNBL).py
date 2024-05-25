import tensorflow as tf2
import tensorflow.compat.v1 as tf
#tf.disable_v2_behavior()

import numpy as np
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.training import moving_averages
import math

def cnn_inference(images, batch_size, n_classes, is_training):
    """
        参数解释：
            images：队列中取的一批图片 [batch_size, width, height, 3]
            batch_size：每个批次的大小
            n_classes：n分类（二分类，猫或狗）
            softmax_linear：表示图片列表中的每张图片分别是猫或狗的预测概率（即：神经网络计算得到的输出值）。
                            一个数值代表属于猫的概率，一个数值代表属于狗的概率，两者的和为 1。
    """

    # 第一层的卷积层conv1，卷积核(weights)的大小是 3*3, 输入的channel(管道数/深度)为3, 共有16个
    with tf.variable_scope('conv1') as scope:

        weights = tf.get_variable('weights',
                                  shape=[3, 3, 3, 16],
                                  dtype=tf.float32,
                                  initializer=tf.truncated_normal_initializer(stddev=0.1, dtype=tf.float32))
        biases = tf.get_variable('biases',
                                 shape=[16],
                                 dtype=tf.float32,
                                 initializer=tf.constant_initializer(0.1))

        conv = tf.nn.conv2d(images, weights, strides=[1, 1, 1, 1], padding='SAME')
        pre_activation = tf.nn.bias_add(conv, biases)

        dim = 128
        reshape = tf.reshape(pre_activation, shape=[1, batch_size * dim, dim, 16])

        ln1 = tf2.keras.layers.LayerNormalization(trainable=is_training)(reshape)

        ln1 = tf.reshape(ln1, shape=[batch_size, dim, dim, 16])

        conv1 = tf.nn.relu(ln1, name='conv1')   # 用relu激活函数非线性化处理

    # 第一层的池化层pool1(特征缩放）
    with tf.variable_scope('pool1') as scope:

        pool1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                               padding='SAME', name='pooling1')

    # 第二层的卷积层cov2，卷积核(weights)的大小是 3*3, 输入的channel(管道数/深度)为16, 共有16个
    with tf.variable_scope('conv2') as scope:
        weights = tf.get_variable('weights',
                                  shape=[3, 3, 16, 16],  # 这里的第三位数字16需要等于上一层的tensor维度
                                  dtype=tf.float32,
                                  initializer=tf.truncated_normal_initializer(stddev=0.1, dtype=tf.float32))
        biases = tf.get_variable('biases',
                                 shape=[16],
                                 dtype=tf.float32,
                                 initializer=tf.constant_initializer(0.1))
        conv = tf.nn.conv2d(pool1, weights, strides=[1, 1, 1, 1], padding='SAME')
        pre_activation = tf.nn.bias_add(conv, biases)

        dim = 64
        reshape = tf.reshape(pre_activation, shape=[1, batch_size * dim, dim, 16])

        ln2 = tf2.keras.layers.LayerNormalization(trainable=is_training)(reshape)

        ln2 = tf.reshape(ln2, shape=[batch_size, dim, dim, 16])

        conv2 = tf.nn.relu(ln2, name='conv2')

    # 第二层的池化层pool2(特征缩放）
    with tf.variable_scope('pool2') as scope:
        pool2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 1, 1, 1],
                               padding='SAME', name='pooling2')

    # 第三层为全连接层local3
    # 连接所有的特征, 将输出值给分类器 (将特征映射到样本标记空间), 该层映射出256个输出
    with tf.variable_scope('local3') as scope:
        # 将pool2张量铺平, 再把维度调整成shape(shape里的-1, 程序运行时会自动计算填充)
        reshape = tf.reshape(pool2, shape=[batch_size, -1])

        dim = reshape.get_shape()[1].value            # 获取reshape后的列数
        weights = tf.get_variable('weights',
                                  shape=[dim, 256],   # 连接256个神经元
                                  dtype=tf.float32,
                                  initializer=tf.truncated_normal_initializer(stddev=0.005, dtype=tf.float32))
        biases = tf.get_variable('biases',
                                 shape=[256],
                                 dtype=tf.float32,
                                 initializer=tf.constant_initializer(0.1))
        # 矩阵相乘再加上biases，用relu激活函数非线性化处理
        local3 = tf.nn.relu(tf.layers.batch_normalization(tf.matmul(reshape, weights) + biases, training=is_training),
                            name='local3')

    # 第四层为全连接层local4
    # 连接所有的特征, 将输出值给分类器 (将特征映射到样本标记空间), 该层映射出512个输出
    with tf.variable_scope('local4') as scope:
        weights = tf.get_variable('weights',
                                  shape=[256, 512],  # 再连接512个神经元
                                  dtype=tf.float32,
                                  initializer=tf.truncated_normal_initializer(stddev=0.005, dtype=tf.float32))
        biases = tf.get_variable('biases',
                                 shape=[512],
                                 dtype=tf.float32,
                                 initializer=tf.constant_initializer(0.1))
        # 矩阵相乘再加上biases，用relu激活函数非线性化处理
        local4 = tf.nn.relu(tf.layers.batch_normalization(tf.matmul(local3, weights) + biases, training=is_training),
                            name='local4')

    # 第五层为输出层(回归层): softmax_linear
    # 将前面的全连接层的输出，做一个线性回归，计算出每一类的得分，在这里是2类，所以这个层输出的是两个得分。
    with tf.variable_scope('softmax_linear') as scope:
        weights = tf.get_variable('weights',
                                  shape=[512, n_classes],
                                  dtype=tf.float32,
                                  initializer=tf.truncated_normal_initializer(stddev=0.005, dtype=tf.float32))
        biases = tf.get_variable('biases',
                                 shape=[n_classes],
                                 dtype=tf.float32,
                                 initializer=tf.constant_initializer(0.1))

        softmax_linear = tf.add(tf.matmul(local4, weights), biases, name='softmax_linear')

    # 这里没做归一化和交叉熵。真正的softmax函数放在下面的losses()里面和交叉熵结合在一起了，提高运算速度。

    return softmax_linear


def losses(logits, labels):
    """
        参数解释：
            logits: 经过cnn_inference得到的神经网络输出值（图片列表中每张图片分别是猫或狗的预测概率）
            labels: 图片对应的标签（即：真实值。用于与logits预测值进行对比得到loss）
            loss： 损失值（label真实值与神经网络输出预测值之间的误差）
    """
    with tf.variable_scope('loss') as scope:
        # label与神经网络输出层的输出结果做对比，得到损失值（在这做归一化和交叉熵处理）
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels, name='loss_per_eg')
        loss = tf.reduce_mean(cross_entropy, name='loss')  # 求得batch的平均loss（每批有batch_size张图）
    return loss



def training(loss, learning_rate):
    """
        参数解释：
            loss: 训练中得到的损失值
            learning_rate：学习率
            train_op: 训练的最优值。训练op，
    """
    with tf.name_scope('optimizer'):
        # 除了利用反向传播算法对权重和偏置项进行修正外，也在运行中不断修正学习率。
        # 根据其损失量学习自适应，损失量大则学习率越大，进行修正的幅度也越大;
        #                     损失量小则学习率越小，进行修正的幅度也越小，但是不会超过自己所设定的学习率。

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):  # 保证train_op在update_ops执行之后再执行。

            optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)    # 使用AdamOptimizer优化器来使loss朝着变小的方向优化

            global_step = tf.Variable(0, name='global_step', trainable=False)

            # global_step：梯度下降一次加1，一般用于记录迭代优化的次数，主要用于参数输出和保存
            train_op = optimizer.minimize(loss, global_step=global_step)   # 以最大限度地最小化loss

    return train_op


def evaluation(logits, labels):
    """
        参数解释：
            logits: 经过cnn_inference得到的神经网络输出值（图片列表中每张图片分别是猫或狗的预测概率）
            labels: 图片对应的标签（真实值，0或1）
            accuracy：准确率（当前step的平均准确率。即：这些batch中多少张图片被正确分类了）
    """
    with tf.variable_scope('accuracy') as scope:
        correct = tf.nn.in_top_k(logits, labels, 1)
        correct = tf.cast(correct, tf.float16)
        accuracy = tf.reduce_mean(correct)            # 计算当前批的平均准确率
    return accuracy
