#-*- coding:utf-8 -*-

import numpy as np
import tensorflow as tf
from custom_VGG import Vgg19
import utils

#loading images
photo = utils.load_image("./Images/argentine.jpg")
art   = utils.load_image("./Images/nuit_etoilee.jpg")

#resizing images
photo_batch = photo.reshape((1,224,224,3))
art_batch   = art.reshape((1,224,224,3))

#creating a noisy output
#output = np.random.rand((1,224,224,3))

with tf.device('/cpu:0'):

    # session : get for the painting and the photo the features from convX.1
    sess = tf.Session()
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

    if not photo_features:
        print("no photo")
    if not art_features:
        print("no art")
