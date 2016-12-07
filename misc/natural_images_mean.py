# -*- coding:utf-8 -*-

import glob
import utils
from numpy.linalg import norm

"""
Script to compute the mean over a big set of natural images in order to
obtain de sigma coefficient for the objective function as defined in
A. Mahendran, A. Vedaldi :Understanding Deep Image Representations by
Inverting Them, BMCV 2014
"""

files = glob.glob("./data/*.jpg")
mean_img = 0.
compt = 0.

for filename in files:
    compt += 1
    img = utils.load_image(filename)
    norm_img = norm(img)
    mean_img = (mean_img * (compt-1) + norm_img) / compt

print("Mean of dataset : {}".format(mean_img))
