#-*- coding:utf-8 -*-

import numpy as np
import tensorflow as tf
from custom_VGG import Vgg19
import utils

#loading images
photo = utils.load_image("./Images/argetine.jpg")
art   = utils.load_image("./Images/nuit_etoilee.jpg")

#resizing images
photo_batch = photo.reshape((1,224,224,3))
art_batch   = art.reshape((1,224,224,3))

#creating a noisy output
output = np.random.rand((1,224,224,3))

with tf.device('/cpu:0'):

    # session : get photography features from convX.1
    sess = tf.Session()
    image = tf.placeholder(tf.float32, [1, 224, 224, 3])
    feed_dict = {image: photo_batch}

    # creating the CNN
    npy_path = 'tensorflow-vgg/'
    cnn = Vgg19(npy_path)
    with tf.name_scope("photo"):
        cnn.build(image)

    # dictionnary of fetches
    cnn_fetches = {p_conv1 = cnn.conv1_1, p_conv2 = cnn.conv2_1, \
                     p_conv3 = cnn.conv3_1, p_conv4 = cnn.conv4.1, \
                     conv5 = cnn.conv5_1}

    # running a session for photography features extraction
    photo_fetches = sess.run(fetches, feed_dict = feed_dict)

    # running a session for art piece features extraction
    art_fetches = sess.run(fetches, feed_dict = feed_dict)
