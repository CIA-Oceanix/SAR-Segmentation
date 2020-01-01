import os
import sys

from keras.models import Model
from keras.optimizers import SGD
from keras.regularizers import l2
from keras.layers import Input, Dense, Dropout, Flatten

from Rignak_Misc.path import get_local_file
from Rignak_DeepLearning.models import convolution_block
from Rignak_DeepLearning.config import get_config

WEIGHT_ROOT = get_local_file(__file__, os.path.join('..', '_outputs', 'models'))
SUMMARY_ROOT = get_local_file(__file__, os.path.join('..', '_outputs', 'summary'))

LOAD = False

LEARNING_RATE = 10 ** -4
WEIGHT_DECAY = 10 ** -5

CONFIG_KEY = 'categorizer'
CONFIG = get_config()[CONFIG_KEY]
DEFAULT_NAME = CONFIG.get('NAME', 'DEFAULT_MODEL_NAME')


def import_model(output_canals, weight_root=WEIGHT_ROOT, summary_root=SUMMARY_ROOT, load=LOAD,
                 weight_decay=WEIGHT_DECAY,
                 learning_rate=LEARNING_RATE, name=DEFAULT_NAME, config=CONFIG):
    weight_filename = os.path.join(weight_root, f"{name}.h5")
    summary_filename = os.path.join(summary_root, f"{name}.txt")

    conv_layers = config['CONV_LAYERS']
    dense_layers = config['DENSE_LAYERS']
    input_shape = config.get('INPUT_SHAPE', (256, 256, 3))
    activation = config.get('ACTIVATION', 'relu')
    last_activation = config.get('LAST_ACTIVATION', 'softmax')

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
    block = Dense(output_canals, activation=last_activation)(block)

    model = Model(inputs=input_layer, outputs=block)

    sgd = SGD(lr=learning_rate, momentum=0.5, nesterov=True)
    model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])

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
