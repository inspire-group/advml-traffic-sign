import keras
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.layers.core import Activation, Dense, Dropout, Flatten
from keras.layers.normalization import BatchNormalization
from keras.models import Sequential
from parameters import *

#----------------------------------- Model ------------------------------------#


def output_fn(correct, predicted):
    return tf.nn.softmax_cross_entropy_with_logits(labels=correct,
                                                   logits=predicted)


def build_mltscl():
    """
    Build multiscale CNN. The last layer must be logits instead of softmax.
    Return a compiled Keras model.
    """

    # Regularization
    l2_reg = keras.regularizers.l2(L2_LAMBDA)

    # Build model
    inpt = keras.layers.Input(shape=IMG_SHAPE)
    conv1 = keras.layers.Convolution2D(
        32, (5, 5), padding='same', activation='relu')(inpt)
    drop1 = keras.layers.Dropout(rate=0.1)(conv1)
    pool1 = keras.layers.MaxPooling2D(pool_size=(2, 2))(drop1)

    conv2 = keras.layers.Convolution2D(
        64, (5, 5), padding='same', activation='relu')(pool1)
    drop2 = keras.layers.Dropout(rate=0.2)(conv2)
    pool2 = keras.layers.MaxPooling2D(pool_size=(2, 2))(drop2)

    conv3 = keras.layers.Convolution2D(
        128, (5, 5), padding='same', activation='relu')(pool2)
    drop3 = keras.layers.Dropout(rate=0.3)(conv3)
    pool3 = keras.layers.MaxPooling2D(pool_size=(2, 2))(drop3)

    pool4 = keras.layers.MaxPooling2D(pool_size=(4, 4))(pool1)
    pool5 = keras.layers.MaxPooling2D(pool_size=(2, 2))(pool2)

    flat1 = keras.layers.Flatten()(pool4)
    flat2 = keras.layers.Flatten()(pool5)
    flat3 = keras.layers.Flatten()(pool3)

    merge = keras.layers.Concatenate(axis=-1)([flat1, flat2, flat3])
    dense1 = keras.layers.Dense(1024, activation='relu',
                                kernel_regularizer=l2_reg)(merge)
    drop4 = keras.layers.Dropout(rate=0.5)(dense1)
    output = keras.layers.Dense(
        OUTPUT_DIM, activation=None, kernel_regularizer=l2_reg)(drop4)
    model = keras.models.Model(inputs=inpt, outputs=output)

    # Specify optimizer
    adam = keras.optimizers.Adam(
        lr=LR, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    model.compile(optimizer=adam, loss=output_fn, metrics=['accuracy'])

    return model


def build_cnn():
    """
    Build CNN. The last layer must be logits instead of softmax.
    Return a compiled Keras model.
    """

    l2_reg = keras.regularizers.l2(L2_LAMBDA)

    # Build model
    inpt = keras.layers.Input(shape=IMG_SHAPE)
    conv1 = keras.layers.Convolution2D(
        32, (5, 5), padding='same', activation='relu')(inpt)
    drop1 = keras.layers.Dropout(rate=0.1)(conv1)
    conv2 = keras.layers.Convolution2D(
        32, (5, 5), padding='same', activation='relu')(drop1)
    drop2 = keras.layers.Dropout(rate=0.2)(conv2)
    pool1 = keras.layers.MaxPooling2D(pool_size=(2, 2))(drop2)

    conv3 = keras.layers.Convolution2D(
        64, (5, 5), padding='same', activation='relu')(pool1)
    drop3 = keras.layers.Dropout(rate=0.3)(conv3)
    conv4 = keras.layers.Convolution2D(
        64, (5, 5), padding='same', activation='relu')(drop3)
    drop4 = keras.layers.Dropout(rate=0.3)(conv4)
    pool2 = keras.layers.MaxPooling2D(pool_size=(2, 2))(drop4)

    flat = keras.layers.Flatten()(pool2)
    dense1 = keras.layers.Dense(200, activation='relu',
                                kernel_regularizer=l2_reg)(flat)
    drop5 = keras.layers.Dropout(rate=0.5)(dense1)
    dense2 = keras.layers.Dense(200, activation='relu',
                                kernel_regularizer=l2_reg)(drop5)
    drop6 = keras.layers.Dropout(rate=0.5)(dense2)
    output = keras.layers.Dense(
        OUTPUT_DIM, activation=None, kernel_regularizer=l2_reg)(drop6)
    model = keras.models.Model(inputs=inpt, outputs=output)

    # Specify optimizer
    adam = keras.optimizers.Adam(
        lr=LR, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    model.compile(optimizer=adam, loss=output_fn, metrics=['accuracy'])

    return model

#---------------------------------- Utility -----------------------------------#


def gradient_fn(model):
    """Return gradient function of model's loss w.r.t. input"""

    x_in = K.placeholder(dtype=tf.float32, shape=INPUT_SHAPE)
    y_true = K.placeholder(shape=(OUTPUT_DIM, ))
    loss = K.categorical_crossentropy(y_true, model(x_in)[0], from_logits=True)
    grad = K.gradients(loss, x_in)

    return K.function([x_in, y_true, K.learning_phase()], grad)


def gradient_input(grad_fn, x, y):
    """Wrapper function to use gradient function more cleanly"""

    return grad_fn([x.reshape(INPUT_SHAPE), y, 0])[0][0]
