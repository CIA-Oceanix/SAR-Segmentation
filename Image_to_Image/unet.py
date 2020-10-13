import sys
import os

from keras.models import Model
from keras.layers import Input, Conv2D
from keras_radam.training import RAdamOptimizer
import keras.backend as K

from Rignak_Misc.path import get_local_file
from Rignak_DeepLearning.models import convolution_block, deconvolution_block, write_summary
from Rignak_DeepLearning.config import get_config
from Rignak_DeepLearning.loss import dice_coef_loss, weighted_binary_crossentropy, get_metrics

WEIGHT_ROOT = get_local_file(__file__, os.path.join('..', '_outputs', 'models'))
SUMMARY_ROOT = get_local_file(__file__, os.path.join('..', '_outputs', 'summary'))
LOAD = False
LEARNING_RATE = 10 ** -5

CONFIG_KEY = 'segmenter'
CONFIG = get_config()[CONFIG_KEY]
DEFAULT_NAME = CONFIG.get('NAME', 'DEFAULT_MODEL_NAME')


def get_additional_metrics(metrics, loss, labels):
    if 'all' in metrics:
        metrics.pop(metrics.index('all'))
        names = [f"{label}_loss" for label in labels]
        metrics += get_metrics(loss, names)
    return metrics


def import_model(weight_root=WEIGHT_ROOT, summary_root=SUMMARY_ROOT, load=LOAD, learning_rate=LEARNING_RATE,
                 config=CONFIG, name=DEFAULT_NAME, skip=True, metrics=None, labels=None):
    convs = []
    block = None

    learning_rate = config.get('LEARNING_RATE', learning_rate)
    batch_normalization = config.get('BATCH_NORMALIZATION', False)
    conv_layers = config['CONV_LAYERS']
    input_shape = config.get('INPUT_SHAPE', (512, 512, 3))
    output_shape = config.get('OUTPUT_SHAPE', (32, 32, 1))
    activation = config.get('ACTIVATION', 'relu')
    if activation == 'sin':
        activation = K.sin
    last_activation = config.get('LAST_ACTIVATION', 'sigmoid')
    loss = config.get('LOSS', 'mse')
    output_canals = output_shape[-1]
    if len(labels) != output_canals:
        labels = [f'class_{i}' for i in range(output_canals)]

    inputs = Input(input_shape)
    # encoder
    for neurons in conv_layers:
        if block is None:
            block, conv = convolution_block(inputs, neurons, activation=activation, maxpool=True,
                                            batch_normalization=batch_normalization)
        else:
            block, conv = convolution_block(block, neurons, activation=activation, maxpool=True,
                                            batch_normalization=batch_normalization)
        convs.append(conv)

    # central
    central_shape = config.get('CENTRAL_SHAPE', conv_layers[-1] * 2)
    block, conv = convolution_block(block, central_shape, activation=activation, maxpool=False,
                                    batch_normalization=batch_normalization)

    # decoder
    for neurons, previous_conv in zip(conv_layers[::-1], convs[::-1]):
        if block.shape[1] == output_shape[0] and block.shape[2] == output_shape[1]:
            break
        if skip:
            block = deconvolution_block(block, previous_conv, neurons, activation=activation)
        else:
            block = deconvolution_block(block, None, neurons, activation=activation)
    conv_layer = Conv2D(output_canals, (1, 1), activation=last_activation)(block)

    model = Model(inputs=[inputs], outputs=[conv_layer])
    optimizer = RAdamOptimizer(learning_rate)

    if loss == "DICE":
        loss = dice_coef_loss
    if loss == "WBCE":
        loss = weighted_binary_crossentropy
    if 'all' in metrics and output_canals != 1:
        names = [f"{label}_loss" for label in labels]
        metrics = get_metrics(loss, names)

    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    model.name = name
    model.summary_filename = os.path.join(summary_root, f"{name}.txt")
    model.weight_filename = os.path.join(weight_root, f"{name}.h5")

    if load:
        print('load weights')
        model.load_weights(model.weight_filename)

    write_summary(model)
    return model
