import sys
import os

from keras.optimizers import Adam
from keras.models import Model
from keras.layers import Input, Conv2D

from Rignak_Misc.path import get_local_file
from Rignak_DeepLearning.models import convolution_block, deconvolution_block
from Rignak_DeepLearning.config import get_config

WEIGHT_ROOT = get_local_file(__file__, os.path.join('..', '_outputs', 'models'))
SUMMARY_ROOT = get_local_file(__file__, os.path.join('..', '_outputs', 'summary'))
LOAD = False
LEARNING_RATE = 10 ** -4

CONFIG_KEY = 'autoencoder'
CONFIG =get_config()[CONFIG_KEY]


def import_model(weight_root=WEIGHT_ROOT, summary_root=SUMMARY_ROOT, load=LOAD, learning_rate=LEARNING_RATE,
                 config=CONFIG, name=CONFIG['NAME']):
    weight_filename = os.path.join(weight_root, f"{name}.h5")
    summary_filename = os.path.join(summary_root, f"{name}.txt")
    convs = []
    block = None

    inputs = Input(config['INPUT_SHAPE'])
    # encoder
    for neurons in config['CONV_LAYERS']:
        if block is None:
            block, conv = convolution_block(inputs, neurons, activation=config['ACTIVATION'], maxpool=True)
        else:
            block, conv = convolution_block(block, neurons, activation=config['ACTIVATION'], maxpool=True)
        convs.append(conv)

    # central
    block, conv = convolution_block(block, config['CONV_LAYERS'][-1] * 2, activation=config['ACTIVATION'],
                                    maxpool=False)

    # decoder
    for neurons, previous_conv in zip(config['CONV_LAYERS'][::-1], convs[::-1]):
        block = deconvolution_block(block, previous_conv, neurons, activation=config['ACTIVATION'])
    conv_layer = Conv2D(config['OUTPUT_CANALS'], (1, 1), activation=config['LAST_ACTIVATION'])(block)

    model = Model(inputs=[inputs], outputs=[conv_layer])
    model.compile(optimizer=Adam(lr=learning_rate), loss='mse')

    model.name = name
    model.weight_filename = weight_filename
    if load:
        print('load weights')
        model.load_weights(weight_filename)

    with open(summary_filename, 'w') as file:
        old = sys.stdout
        sys.stdout = file
        model.summary()
        sys.stdout = old

    return model
