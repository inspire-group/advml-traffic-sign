"""
Define project-wide parameters in this 'configuration' file
"""

# Import packages for all files
import os
import pickle
import random
import threading
import time
from os import listdir

import cv2
import keras
import keras.backend as K
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from keras.optimizers import SGD
from keras.preprocessing.image import ImageDataGenerator
from pandas.io.parsers import read_csv
from scipy import misc
from tensorflow.contrib.opt import ScipyOptimizerInterface

# Set constants (MNIST)
# NUM_LABELS = 10                             # Number of labels
# BATCH_SIZE = 32                             # Size of batch
# HEIGHT = 28                                 # Height of input image
# WIDTH = 28                                  # Width of input image
# N_CHANNEL = 1                               # Number of channels
# OUTPUT_DIM = 10                             # Number of output dimension

# Set constants (GTSRB)
NUM_LABELS = 43                             # Number of labels
BATCH_SIZE = 32                             # Size of batch
HEIGHT = 32                                 # Height of input image
WIDTH = 32                                  # Width of input image
N_CHANNEL = 3                               # Number of channels
OUTPUT_DIM = 43                             # Number of output dimension

# Set training hyperparameters
NUM_EPOCH = 100                             # Number of epoch to train
LR = 0.0001                                 # Learning rate
L2_LAMBDA = 0.0001                          # Lambda for l2 regularization

# Set paths
# Path to saved weights
WEIGTHS_PATH = "./keras_weights/weights_mltscl_dataaug.hdf5"
# Path to directory containing dataset
DATA_DIR = "./input_data/"

INPUT_SHAPE = (1, HEIGHT, WIDTH, N_CHANNEL)  # Input shape of model
IMG_SHAPE = (HEIGHT, WIDTH, N_CHANNEL)
IMAGE_SIZE = (HEIGHT, WIDTH)                # Height and width of resized image
N_FEATURE = HEIGHT * WIDTH * N_CHANNEL      # Number of input dimension
