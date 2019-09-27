import sys
import os

from keras.optimizers import Adam
from keras.models import Model
from keras.layers import Input, concatenate, Conv2D

from Rignak_Misc.path import get_local_file
from Rignak_DeepLearning.models import convolution_block

WEIGHT_ROOT = get_local_file(__file__, os.path.join('..', '_outputs', 'models'))
SUMMARY_ROOT = get_local_file(__file__, os.path.join('..', '_outputs', 'summary'))
NAME = 'flat_autoencoder'
INPUT_SHAPE = (256, 256)
CONV_LAYERS = (4, 8, 16, 32)
OUTPUT_CANALS = 3
NEURON_BASIS = 8
ACTIVATION = 'relu'
LAST_ACTIVATION = 'sigmoid'
LOAD = False


def import_model(weight_root=WEIGHT_ROOT, input_shape=INPUT_SHAPE, canals=OUTPUT_CANALS,
                 conv_layers=CONV_LAYERS, activation=ACTIVATION, last_activation=LAST_ACTIVATION,
                 summary_root=SUMMARY_ROOT, load=LOAD, name=NAME):
    weight_filename = os.path.join(weight_root, f"{name}.h5")
    summary_filename = os.path.join(summary_root, f"{name}.txt")
    convs = []
    block = None

    inputs = Input(input_shape)
    # encoder
    for neurons in conv_layers:
        if block is None:
            block, conv = convolution_block(inputs, neurons, activation=activation, maxpool=False)
        else:
            block, conv = convolution_block(block, neurons, activation=activation, maxpool=False)
        convs.append(convs)

    # central
    block, conv = convolution_block(block, conv_layers[-1]*2, activation=activation, maxpool=False)

    # decoder
    for neurons, previous_conv in zip(conv_layers[::-1], convs[::-1]):
        concatenation = concatenate([block, previous_conv], axis=3)
        block, _ = convolution_block(concatenation, neurons, activation=activation, maxpool=False)

    outputs = Conv2D(canals, (1, 1), activation=last_activation)(block)

    model = Model(inputs=[inputs], outputs=[outputs])
    model.compile(optimizer=Adam(lr=1e-4), loss='binary_crossentropy')

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
