#-*- coding:utf-8 -*-

import numpy as np
import tensorflow as tf
from custom_VGG import Vgg19
from tensor_function import *
import utils

#loading images
photo = utils.load_image("./Images/argentine.jpg")
art   = utils.load_image("./Images/nuit_etoilee.jpg")

#resizing images
photo_batch = photo.reshape((1,224,224,3))
art_batch   = art.reshape((1,224,224,3))

#creating a noisy output
output = np.random.rand(1,224,224,3).astype(np.float32)

#ratios structure vs style
alpha = 1
beta = 1

with tf.device('/cpu:0'):

    # session : get for the painting and the photo the features from convX.1
    with tf.Session() as sess:
        image = tf.placeholder(tf.float32, [1, 224, 224, 3])
        photo_dict = {image: photo_batch}
        art_dict   = {image: art_batch}

        # creating the CNN
        cnn = Vgg19('./vgg19.npy')
        with tf.name_scope("photo"):
            cnn.build(image)

        # dictionnary of fetches
        fetches = (cnn.conv1_1, cnn.conv2_1, cnn.conv3_1, cnn.conv4_1, \
                       cnn.conv5_1)

        # running a session for photography featuresextraction
        photo_features = sess.run(fetches, feed_dict = photo_dict)

        # running a session for art piece features extraction
        art_features = sess.run(fetches, feed_dict = art_dict)

        # running a session for the output
        out_dict = {image : output}
        out_features = sess.run(fetches, feed_dict = out_dict)

        # losses for each layer
        photo_output_distance = [tensor_distance(out_features[i], \
                                photo_features[i], True) for i in np.arange(5)]
        art_output_distance = [tensor_distance(out_features[i], \
                                art_features, True) for i in np.arange(5)]

        content_loss = [tf.reduce_mean(photo_output_distance[i]) \
                        for i in np.arange(5)]
        style_loss = [tf.reduce_mean(art_output_distance[i] \
                        for i in np.arange(5)]

        total_loss = alpha*content_loss + beta*style_loss
