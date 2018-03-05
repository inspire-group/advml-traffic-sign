"""
A run script for training NN model on GTSRB
"""

import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from lib.keras_utils import *
from lib.OptCarlini import *
from lib.OptTransform import *
from lib.RandomTransform import *
from lib.utils import *
from parameters import *

# Set training metadata
LOAD_WEIGHTS = False
LOAD_WEIGHTS_PATH = './weights.24-0.20.hdf5'
TRAIN_FILE_NAME = 'train_extended_75.p'

# Build model. model is a compiled Keras model with last layer being logits.
model = build_mltscl()
if LOAD_WEIGHTS:
    model.load_weights(LOAD_WEIGHTS_PATH)

# Load dataset
x_train, y_train, x_val, y_val, x_test, y_test = load_dataset_GTSRB(
    n_channel=N_CHANNEL, train_file_name=TRAIN_FILE_NAME)

# Convert to one-hot encoding
y_train = keras.utils.to_categorical(y_train, NUM_LABELS)
y_test = keras.utils.to_categorical(y_test, NUM_LABELS)
y_val = keras.utils.to_categorical(y_val, NUM_LABELS)

# Path to save weights
filepath = './weights.{epoch:02d}-{val_loss:.2f}.hdf5'

# Callback function to save weights every epoch
modelCheckpoint = keras.callbacks.ModelCheckpoint(
    filepath, monitor='val_loss', verbose=0, save_best_only=False,
    save_weights_only=False, mode='auto', period=1)
# Callback function for early stopping
# earlyStop = keras.callbacks.EarlyStopping(
#    monitor='val_loss', min_delta=0.001, patience=5, verbose=0, mode='auto')

model.fit(x_train, y_train, batch_size=BATCH_SIZE, epochs=NUM_EPOCH, verbose=1,
          callbacks=[modelCheckpoint], validation_data=(
              x_val, y_val),
          shuffle=True, initial_epoch=0)
