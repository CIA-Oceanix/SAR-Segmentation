from keras.layers import Conv2D, MaxPooling2D, Conv2DTranspose, concatenate, BatchNormalization, Dropout

KERNEL_SIZE = (3, 3)
ACTIVATION = 'relu'
DROPOUT = 0.0


def convolution_block(layer, neurons, kernel_size=KERNEL_SIZE, activation=ACTIVATION,
                      maxpool=True, batch_normalization=False, block_depth=3, use_bias=True):
    for _ in range(block_depth):
        layer = Conv2D(neurons, kernel_size, activation=activation, padding='same', use_bias=use_bias)(layer)
    if maxpool:
        block = MaxPooling2D(pool_size=(2, 2))(layer)
    else:
        block = layer
    if batch_normalization:
        block = BatchNormalization()(block)
    return block, layer


def deconvolution_block(layer, previous_conv, neurons, kernel_size=KERNEL_SIZE, activation=ACTIVATION,
                        dropout=DROPOUT, batch_normalization=False, block_depth=3, use_bias=True):
    block = Conv2DTranspose(neurons, kernel_size, strides=(2, 2), padding='same', use_bias=use_bias)(layer)
    if previous_conv is not None:
        previous_conv = Dropout(dropout)(previous_conv)
        block = concatenate([block, previous_conv], axis=3)
    for _ in range(block_depth):
        block = Conv2D(neurons, kernel_size, activation=activation, padding='same', use_bias=use_bias)(block)
    if batch_normalization:
        block = BatchNormalization()(block)
    return block
