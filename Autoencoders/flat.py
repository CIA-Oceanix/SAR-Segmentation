import sys
import os

from keras.optimizers import Adam
from keras.models import Model, load_model
from keras.layers import Input, Conv2D, Dropout, concatenate

from Rignak_Misc.path import get_local_file
from Rignak_DeepLearning.models import convolution_block
from Rignak_DeepLearning.config import get_config

WEIGHT_ROOT = get_local_file(__file__, os.path.join('..', '_outputs', 'models'))
SUMMARY_ROOT = get_local_file(__file__, os.path.join('..', '_outputs', 'summary'))
LOAD = False

LEARNING_RATE = 10 ** -4

CONFIG_KEY = 'saliency'
CONFIG = get_config()[CONFIG_KEY]

DROPOUT = 0.

def import_model(weight_root=WEIGHT_ROOT, summary_root=SUMMARY_ROOT, load=LOAD, learning_rate=LEARNING_RATE,
                 name=CONFIG['NAME'], config=CONFIG, dropout =DROPOUT):
    weight_filename = os.path.join(weight_root, f"{name}.h5")
    summary_filename = os.path.join(summary_root, f"{name}.txt")
    convs = []
    block = None

    pretrained_model_filename = os.path.join(weight_root, f"{name.replace('saliency', 'flat_autoencoder')}.h5")
    if 'saliency' in name:
        print('Will search for a pretrained encoder')
    if 'saliency' in name and os.path.exists(pretrained_model_filename):
        print('Found a flat_autoencoder model, will proceed to load it')
        model = load_model(pretrained_model_filename)
        for layer in model.layers[:len(config['CONV_LAYERS'])*3 + 1]:
            layer.trainable = False
        inputs = model.input
        block = model.layers[-2].output
    else:
        inputs = Input(config['INPUT_SHAPE'])
        # encoder
        for neurons in config['CONV_LAYERS']:
            if block is None:
                block, conv = convolution_block(inputs, neurons, activation=config['ACTIVATION'], maxpool=False)
            else:
                block, conv = convolution_block(block, neurons, activation=config['ACTIVATION'], maxpool=False)
            convs.append(conv)

        # central
        block, conv = convolution_block(block, config['CONV_LAYERS'][-1] * 2, activation=config['ACTIVATION'],
                                        maxpool=False)

        # decoder
        for neurons, previous_conv in zip(config['CONV_LAYERS'][::-1], convs[::-1]):
            previous_conv = Dropout(dropout)(previous_conv)
            block = concatenate([block, previous_conv], axis=3)
            block, _ = convolution_block(block, neurons, activation=config['ACTIVATION'], maxpool=False)

    outputs = Conv2D(config['OUTPUT_CANALS'], (1, 1), activation=config['LAST_ACTIVATION'], name='output')(block)

    model = Model(inputs=[inputs], outputs=[outputs])
    model.compile(optimizer=Adam(lr=learning_rate), loss='binary_crossentropy')

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
