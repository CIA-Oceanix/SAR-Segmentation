import os
import sys

from keras.models import Model
from keras.optimizers import SGD, Adam
from keras.regularizers import l2
from keras.layers import Input, Conv2D, Dense, MaxPooling2D, GaussianNoise, Dropout, Flatten

from Rignak_Misc.path import get_local_file
from Rignak_DeepLearning.models import convolution_block

WEIGHT_ROOT = get_local_file(__file__, os.path.join('..', '_outputs', 'models'))
SUMMARY_ROOT = get_local_file(__file__, os.path.join('..', '_outputs', 'summary'))
NAME = 'flat_categorizer'
INPUT_SHAPE = (150, 150, 3)
CONV_LAYERS = (64, 128, 256, 256)
DENSE_LAYERS = (512, 256)
ACTIVATION = 'relu'
LAST_ACTIVATION = 'softmax'

LOAD = False

LEARNING_RATE = 10 ** -4
WEIGHT_DECAY = 10 ** -5


def import_model(canals, weight_root=WEIGHT_ROOT, input_shape=INPUT_SHAPE, activation=ACTIVATION,
                 summary_root=SUMMARY_ROOT, load=LOAD, name=NAME, conv_layers=CONV_LAYERS,
                 dense_layers=DENSE_LAYERS, weight_decay=WEIGHT_DECAY, learning_rate=LEARNING_RATE,
                 last_activation=LAST_ACTIVATION):
    weight_filename = os.path.join(weight_root, f"{name}.h5")
    summary_filename = os.path.join(summary_root, f"{name}.txt")

    input_layer = Input(input_shape)

    block = None
    for neurons in conv_layers:
        if block is None:
            block, _ = convolution_block(input_layer, neurons, activation=activation)
        else:
            block, _ = convolution_block(block, neurons, activation=activation)

    block = Flatten()(block)
    for neurons in dense_layers:
        block = Dense(neurons, kernel_regularizer=l2(weight_decay), activation=activation)(block)
        block = Dropout(0.5)(block)
    block = Dense(canals, activation=last_activation)(block)

    model = Model(inputs=input_layer, outputs=block)

    sgd = SGD(lr=learning_rate, momentum=0.5, nesterov=True)
    model.compile(optimizer=sgd, loss='mse', metrics=['accuracy'])

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
