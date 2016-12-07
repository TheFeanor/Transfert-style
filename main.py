#-*- coding:utf-8 -*-

import numpy as np
import tensorflow as tf
from custom_VGG import Vgg19
from tensor_functions import *
import utils
import skimage.io
from time import time

#loading images
photo = utils.load_image("./Images/argentine.jpg")
art   = utils.load_image("./Images/nuit_etoilee.jpg")

#resizing images
photo_batch = np.reshape(photo, (1, 224, 224, 3))
art_batch   = np.reshape(art, (1, 224, 224, 3))

#ratios structure vs style
alpha = 1
beta = 1

#layer where to exctract features_a
layer = 5

with tf.device('/cpu:0'):

    # session : get for the painting and the photo the features from convX.1
    with tf.Session() as sess:
        # running a session for photography features extraction
        image = tf.convert_to_tensor(photo_batch, dtype=tf.float32)
        cnn = Vgg19('./vgg19.npy')
        cnn.build(image)
        photo_features = [cnn.conv1_1, cnn.conv2_1, cnn.conv3_1, cnn.conv4_1, \
                       cnn.conv5_1]
        print("Features extraction from photography, done !")

        # running a session for art piece features extraction
        image = tf.convert_to_tensor(art_batch, dtype=tf.float32)
        cnn = Vgg19('./vgg19.npy')
        cnn.build(image)
        art_features = [cnn.conv1_1, cnn.conv2_1, cnn.conv3_1, cnn.conv4_1, \
                       cnn.conv5_1]
        print("Features extraction from art piece, done !")

        # running a session for the output
        output = tf.Variable(255*tf.random_uniform([1,224,224,3]))

        # initialization of all the variables
        init = tf.initialize_all_variables()
        sess.run(init)
        #print(output.eval()[:,:10,:10,0])
        cnn = Vgg19('./vgg19.npy')
        cnn.build(output)
        out_features = [cnn.conv1_1, cnn.conv2_1, cnn.conv3_1, cnn.conv4_1, \
                       cnn.conv5_1]
        print("Features extraction from noise, done !")

        # losses for each layer
        start = time()
        #style_loss = style_error(art_features, out_features)
        #alpha_loss = alpha_reg(x=output, alpha=6, lambd=2.16e8)
        #beta_loss = TV_reg(x=output, beta=2, lambd=5)
        #total_loss = style_loss #+ alpha_loss + beta_loss
        total_loss = []
        for k in np.arange(5):
            total_loss.append(structure_error(photo_features, \
                                                  out_features, k))

        sess.run(total_loss)
        end = time()
        print("Loss computation, done in {} s!".format(end-start))

        print(tf.trainable_variables())

        # minimization of the loss
        l_rate = 0.000001
        decay = 0.9
        opt = tf.train.GradientDescentOptimizer(learning_rate = l_rate)
        loss = total_loss[layer-1]
        train = opt.minimize(loss)
        print("Gradient descent construction, done !")

        # let's begin !
        sess.run(tf.initialize_all_variables())
        for step in np.arange(10):
            sess.run(train)
            if (step%10 == 0):
                print("Optimization step : {}".format(step))

        # display the result
        image_out = np.squeeze(sess.run(output))
        print(image_out[:10,:10,:])
        skimage.io.imsave("./images/out/output.jpg", image_out)
