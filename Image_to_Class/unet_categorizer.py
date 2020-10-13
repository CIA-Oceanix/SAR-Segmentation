import os
import sys

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import Rignak_DeepLearning.deprecation_warnings

from keras.models import Model
from keras.layers import Input, GlobalAveragePooling2D, Dense
from keras.optimizers import Adam
from keras_radam.training import RAdamOptimizer
import runai.ga.keras

from Rignak_Misc.path import get_local_file
from Rignak_DeepLearning.config import get_config
from Rignak_DeepLearning.models import convolution_block, deconvolution_block, write_summary

WEIGHT_ROOT = get_local_file(__file__, os.path.join('..', '_outputs', 'models'))
SUMMARY_ROOT = get_local_file(__file__, os.path.join('..', '_outputs', 'summary'))
LOAD = False
LEARNING_RATE = 10 ** -4

CONFIG_KEY = 'autoencoder'
CONFIG = get_config()[CONFIG_KEY]
DEFAULT_NAME = CONFIG.get('NAME', 'DEFAULT_MODEL_NAME')
GRADIENT_ACCUMULATION = 4
DEFAULT_METRICS = ['accuracy']


def import_model(weight_root=WEIGHT_ROOT, summary_root=SUMMARY_ROOT, load=LOAD, learning_rate=LEARNING_RATE,
                 config=CONFIG, name=DEFAULT_NAME, gradient_accumulation=GRADIENT_ACCUMULATION,
                 metrics=DEFAULT_METRICS):
    convs = []
    block = None

    batch_normalization = False
    conv_layers = config.get('CONV_LAYERS', [32, 64])

    input_shape = config.get('INPUT_SHAPE', (256, 256, 3))
    activation = config.get('ACTIVATION', 'relu')
    last_activation = config.get('LAST_ACTIVATION', 'sigmoid')
    output_canals = config.get('OUTPUT_CANALS')
    block_depth = config.get('BLOCK_DEPTH', 3)

    loss = config.get('LOSS', 'categorical_crossentropy')

    inputs = Input(input_shape)
    # encoder
    for neurons in conv_layers:
        if block is None:
            block, conv = convolution_block(inputs, neurons, activation=activation, maxpool=True,
                                            batch_normalization=batch_normalization, block_depth=block_depth)
        else:
            block, conv = convolution_block(block, neurons, activation=activation, maxpool=True,
                                            batch_normalization=batch_normalization, block_depth=block_depth)
        convs.append(conv)

    # central
    central_nodes = config.get('CENTRAL', conv_layers[-1] * 2)
    block, conv = convolution_block(block, central_nodes, activation=activation, maxpool=False,
                                    batch_normalization=batch_normalization, block_depth=block_depth)

    # decoder
    for neurons, previous_conv in zip(conv_layers[::-1], convs[::-1]):
        block = deconvolution_block(block, previous_conv, neurons, activation=activation, block_depth=block_depth)
    average_layer = GlobalAveragePooling2D()(block)
    output_layer = Dense(output_canals, activation=last_activation, use_bias=False)(average_layer)

    model = Model(inputs=[inputs], outputs=[output_layer])

    optimizer = Adam(learning_rate)
    optimizer = runai.ga.keras.optimizers.Optimizer(optimizer, steps=gradient_accumulation)
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    model.name = name
    model.weight_filename = os.path.join(weight_root, f"{name}.h5")
    model.summary_filename = os.path.join(summary_root, f"{name}.txt")
    if load:
        print('load weights')
        model.load_weights(model.weight_filename)

    write_summary(model)
    return model


if __name__ == '__main__':
    gan = import_model()
