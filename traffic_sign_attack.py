import random
import threading
import time
from os import listdir

import numpy as np
import tensorflow as tf
import keras
from scipy import misc

import matplotlib.pyplot as plt
from pandas.io.parsers import read_csv

from parameters import *
from lib.utils.traffic_sign import *
from lib.utils.keras_utils import *


# Build and load trained model
model = built_mltscl()
model.load_weights(WEIGTHS_PATH)

# Load dataset
x_train, y_train, x_val, y_val, x_test, y_test = load_dataset_GTSRB(
    n_channel=N_CHANNEL)

y_train = keras.utils.to_categorical(y_train, NUM_LABELS)
y_test = keras.utils.to_categorical(y_test, NUM_LABELS)
y_val = keras.utils.to_categorical(y_val, NUM_LABELS)

# Read sign names
signnames = read_csv("./signnames.csv").values[:, 1]

# Create adversarial examples
mag_list = np.linspace(0.2, 2, 10)
x_adv = fg(model, x_test, y_test, mag_list)

print(eval_adv(model, x_adv, y_test))
