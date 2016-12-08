#-*- coding:utf-8 -*-

import numpy as np
import tensorflow as tf
from custom_VGG import Vgg19
from tensor_functions import *
import utils
import skimage.io
from time import time

#loading images
photo = 128*(utils.load_image("./Images/argentine.jpg")-0.5)
art   = 128*(utils.load_image("./Images/nuit_etoilee.jpg")-0.5)

#resizing images
photo_batch = np.reshape(photo, (1, 224, 224, 3))
art_batch   = np.reshape(art, (1, 224, 224, 3))

print(np.amin(photo_batch))

#ratios structure vs style
alpha = 1
beta = 1

#layer where to exctract features_a
layer = 5

# mean of natural images
sigma = 191.7
lambd_a = 2.16e8
#lambd_b = 0.5 # layers 1-6
#lambd_b = 5   # layers 7-12
lambd_b = 50   # layers 13-20

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
        output = tf.Variable(128*tf.random_uniform([1,224,224,3], \
                                    minval = -1, maxval = 1))

        # initialization of all the variables
        init = tf.initialize_all_variables()
        sess.run(init)
        #print(output.eval()[:,:10,:10,0])
        cnn = Vgg19('./vgg19.npy')
        cnn.build(sigma*output) # to compute features of sigma * x'
        out_features = [cnn.conv1_1, cnn.conv2_1, cnn.conv3_1, cnn.conv4_1, \
                       cnn.conv5_1]
        print("Features extraction from noise, done !")

        # losses for each layer
        start = time()
        #style_loss = style_error(art_features, out_features)
        alpha_loss = alpha_reg(x=output, alpha=6, lambd=lambd_a)
        beta_loss = TV_reg(x=output, beta=2, lambd=lambd_b)
        #total_loss = style_loss #+ alpha_loss + beta_loss
        total_loss = []
        for k in np.arange(5):
            total_loss.append(structure_error(photo_features, \
                                                  out_features, k))

        #sess.run(total_loss)
        end = time()
        print("Loss computation, done in {} s!".format(end-start))

        #print(tf.trainable_variables())

        # minimization of the loss
        decay = 0.9
        opt = tf.train.GradientDescentOptimizer(0.5)
        loss = total_loss[layer-1]
        train = opt.minimize(loss)
        print("Gradient descent construction, done !")

        # let's begin !
        sess.run(tf.initialize_all_variables())
        for step in np.arange(100):
            sess.run(train)
            if (step%10 == 0):
                print("Optimization step : {}".format(step))

        # display the result
        image_out = 128+np.squeeze(sess.run(output))
        (row, col, channel) = image_out.shape

        # assure that values are in [0, 255]
        for i in np.arange(row):
            for j in np.arange(col):
                for k in np.arange(channel):
                    if image_out[i, j, k] < 0:
                        image_out[i, j, k] = 0
                    if image_out[i, j, k] > 255:
                        image_out[i, j, k] = 255

        #print(image_out[:10,:10,:])
        print(np.amax(image_out))
        print(np.amin(image_out))
        skimage.io.imsave("./Images/out/output.jpg", image_out/256)
