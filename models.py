
from keras.layers import Conv2D, MaxPooling2D, Conv2DTranspose, concatenate, BatchNormalization


KERNEL_SIZE = (3, 3)
ACTIVATION = 'relu'


def convolution_block(layer, neurons, kernel_size=KERNEL_SIZE, activation=ACTIVATION, maxpool=True):
    layer = Conv2D(neurons, kernel_size, activation=activation, padding='same')(layer)
    layer = Conv2D(neurons, kernel_size, activation=activation, padding='same')(layer)
    layer = Conv2D(neurons, kernel_size, activation=activation, padding='same')(layer)
    if maxpool:
        block = MaxPooling2D(pool_size=(2, 2))(layer)
    else:
        block = layer
    block = BatchNormalization()(block)
    return block, layer


def deconvolution_block(layer, previous_conv, neurons, kernel_size=KERNEL_SIZE, activation=ACTIVATION):
    block = Conv2DTranspose(neurons, kernel_size, strides=(2, 2), padding='same')(layer)
    block = concatenate([block, previous_conv], axis=3)
    block = Conv2D(neurons, kernel_size, activation=activation, padding='same')(block)
    block = Conv2D(neurons, kernel_size, activation=activation, padding='same')(block)
    block = Conv2D(neurons, kernel_size, activation=activation, padding='same')(block)
    block = BatchNormalization()(block)
    return block