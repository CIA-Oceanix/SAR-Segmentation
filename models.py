import os
import sys
import json

from tensorflow.keras.layers import Conv2D, MaxPooling2D, Conv2DTranspose, concatenate, BatchNormalization, Dropout
from tensorflow.keras.models import load_model, model_from_json
 
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


def write_summary(model):
    os.makedirs(os.path.split(model.summary_filename)[0], exist_ok=True)
    with open(model.summary_filename, 'w') as file:
        old = sys.stdout
        sys.stdout = file
        model.summary()
        sys.stdout = old


def load_weights(model, weight_filename, load, freeze):
    if load:
        print('load weights')
        old_model = load_model(weight_filename, compile=False)
        for layer, old_layer in zip(model.layers, old_model.layers):
            try:
                layer.set_weights(old_layer.get_weights())
            except ValueError as e:
                print(e)
    if freeze:
        for i in range(len(model.layers)-1):
            model.layers[i].trainable = False
    return model


def flatten_model(model, axis=0):
    json_model = json.loads(model.to_json())

    for layer in json_model["config"]['layers']:
        config = layer['config']
        if "strides" in config: config["strides"][axis] = 1
        if "kernel_size" in config: config["kernel_size"][axis] = 1
        if "dilation_rate" in config: config['dilation_rate'][axis] = 1
        if "padding" in config: config['padding'] = 'same'

    json_model["config"]['layers'][0]['config']['batch_input_shape'][axis+1] = 1
    json_model["config"]['layers'][0]['config']['batch_input_shape'][3] = 1

    return model_from_json(json.dumps(json_model))
