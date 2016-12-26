import os
import tensorflow as tf

import numpy as np
import time
import inspect

VGG_MEAN = [103.939, 116.779, 123.68]
data_dict = np.load('./vgg19.npy', encoding='latin1').item()

filt_conv1_1 = data_dict["conv1_1"][0]
bias_conv1_1 = data_dict["conv1_1"][1]
filt_conv1_2 = data_dict["conv1_2"][0]
bias_conv1_2 = data_dict["conv1_2"][1]

filt_conv2_1 = data_dict["conv2_1"][0]
bias_conv2_1 = data_dict["conv2_1"][1]
filt_conv2_2 = data_dict["conv2_2"][0]
bias_conv2_2 = data_dict["conv2_2"][1]

filt_conv3_1 = data_dict["conv3_1"][0]
bias_conv3_1 = data_dict["conv3_1"][1]
filt_conv3_2 = data_dict["conv3_2"][0]
bias_conv3_2 = data_dict["conv3_2"][1]
filt_conv3_3 = data_dict["conv3_3"][0]
bias_conv3_3 = data_dict["conv3_3"][1]
filt_conv3_4 = data_dict["conv3_4"][0]
bias_conv3_4 = data_dict["conv3_4"][1]

filt_conv4_1 = data_dict["conv4_1"][0]
bias_conv4_1 = data_dict["conv4_1"][1]
filt_conv4_2 = data_dict["conv4_2"][0]
bias_conv4_2 = data_dict["conv4_2"][1]
filt_conv4_3 = data_dict["conv4_3"][0]
bias_conv4_3 = data_dict["conv4_3"][1]
filt_conv4_4 = data_dict["conv4_4"][0]
bias_conv4_4 = data_dict["conv4_4"][1]

filt_conv5_1 = data_dict["conv5_1"][0]
bias_conv5_1 = data_dict["conv5_1"][1]
filt_conv5_2 = data_dict["conv5_2"][0]
bias_conv5_2 = data_dict["conv5_2"][1]
filt_conv5_3 = data_dict["conv5_3"][0]
bias_conv5_3 = data_dict["conv5_3"][1]
filt_conv5_4 = data_dict["conv5_4"][0]
bias_conv5_4 = data_dict["conv5_4"][1]

data_dict = None

def avg_pool(bottom, name):
    average =  tf.nn.avg_pool(bottom, ksize=[1, 2, 2, 1], \
                        strides=[1, 2, 2, 1], padding='SAME', name=name)
    return average

def max_pool(bottom, name):
    max_ = tf.nn.max_pool(bottom, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)
    return max_

def conv_layer(bottom, filt, conv_biases):
    conv = tf.nn.conv2d(bottom, filt, [1, 1, 1, 1], padding='SAME')
    bias = tf.nn.bias_add(conv, conv_biases)
    relu = tf.nn.relu(bias)

    return relu


def build(rgb):
    """
    load variable from npy to build the VGG

    :param rgb: rgb image [batch, height, width, 3] values scaled [0, 1]
    """

    start_time = time.time()
    #print("build model started")
    rgb_scaled = rgb * 255.0

    # Convert RGB to BGR
    red, green, blue = tf.split(3, 3, rgb_scaled)
    assert red.get_shape().as_list()[1:] == [224, 224, 1]
    assert green.get_shape().as_list()[1:] == [224, 224, 1]
    assert blue.get_shape().as_list()[1:] == [224, 224, 1]
    bgr = tf.concat(3, [
        blue - VGG_MEAN[0],
        green - VGG_MEAN[1],
        red - VGG_MEAN[2],
    ])
    assert bgr.get_shape().as_list()[1:] == [224, 224, 3]

    conv1_1 = conv_layer(bgr, filt_conv1_1, bias_conv1_1)
    conv1_2 = conv_layer(conv1_1, filt_conv1_2, bias_conv1_2)
    pool1 = avg_pool(conv1_2, 'pool1')

    conv2_1 = conv_layer(pool1, filt_conv2_1, bias_conv2_1)
    conv2_2 = conv_layer(conv2_1, filt_conv2_2, bias_conv2_2)
    pool2 = avg_pool(conv2_2, 'pool2')

    conv3_1 = conv_layer(pool2, filt_conv3_1, bias_conv3_1)
    conv3_2 = conv_layer(conv3_1, filt_conv3_2, bias_conv3_2)
    conv3_3 = conv_layer(conv3_2, filt_conv3_3, bias_conv3_3)
    conv3_4 = conv_layer(conv3_3, filt_conv3_4, bias_conv3_4)
    pool3 = avg_pool(conv3_4, 'pool3')

    conv4_1 = conv_layer(pool3, filt_conv4_1, bias_conv4_1)
    conv4_2 = conv_layer(conv4_1, filt_conv4_2, bias_conv4_2)
    conv4_3 = conv_layer(conv4_2, filt_conv4_3, bias_conv4_3)
    conv4_4 = conv_layer(conv4_3, filt_conv4_4, bias_conv4_4)
    pool4 = avg_pool(conv4_4, 'pool4')

    conv5_1 = conv_layer(pool4, filt_conv5_1, bias_conv5_1)
    # conv5_2 = conv_layer(conv5_1, filt_conv5_2, bias_conv5_2)
    # conv5_3 = conv_layer(conv5_2, filt_conv5_3, bias_conv5_3)
    # conv5_4 = conv_layer(conv5_3, filt_conv5_4, bias_conv5_4)
    # pool5 = avg_pool(conv5_4, 'pool5')

    return conv1_1, conv2_1, conv3_1, conv4_1, conv5_1
