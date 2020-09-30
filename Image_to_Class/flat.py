import os
import sys

from keras.models import Model
from keras.optimizers import SGD
from keras.regularizers import l2
from keras.layers import Input, Dense, Dropout, Flatten
from keras.optimizers import Adam
from keras_radam.training import RAdamOptimizer
import runai.ga.keras


from Rignak_Misc.path import get_local_file
from Rignak_DeepLearning.models import convolution_block
from Rignak_DeepLearning.config import get_config

WEIGHT_ROOT = get_local_file(__file__, os.path.join('..', '_outputs', 'models'))
SUMMARY_ROOT = get_local_file(__file__, os.path.join('..', '_outputs', 'summary'))

LOAD = False

LEARNING_RATE = 10 ** -3

CONFIG_KEY = 'categorizer'
CONFIG = get_config()[CONFIG_KEY]
DEFAULT_NAME = CONFIG.get('NAME', 'DEFAULT_MODEL_NAME')
GRADIENT_ACCUMULATION = 8


def import_model(output_canals, weight_root=WEIGHT_ROOT, summary_root=SUMMARY_ROOT, load=LOAD,
                 learning_rate=LEARNING_RATE, name=DEFAULT_NAME, config=CONFIG,
                 class_weight=None, gradient_accumulation=GRADIENT_ACCUMULATION):
    weight_filename = os.path.join(weight_root, f"{name}.h5")
    summary_filename = os.path.join(summary_root, f"{name}.txt")

    conv_layers = config['CONV_LAYERS']
    dense_layers = config['DENSE_LAYERS']
    input_shape = config.get('INPUT_SHAPE', (128, 128, 3))
    activation = config.get('ACTIVATION', 'relu')
    last_activation = config.get('LAST_ACTIVATION', 'softmax')
    block_depth = config.get('BLOCK_DEPTH', 1)

    input_layer = Input(input_shape)
    block = None
    for neurons in conv_layers:
        if block is None:
            block, _ = convolution_block(input_layer, neurons, activation=activation,
                                         batch_normalization=False, block_depth=block_depth)
        else:
            block, _ = convolution_block(block, neurons, activation=activation, batch_normalization=False,
                                         block_depth=block_depth)

    block = Flatten()(block)
    for neurons in dense_layers:
        block = Dense(neurons, activation=activation)(block)
        # block = Dropout(0.5)(block)
    block = Dense(output_canals, activation=last_activation)(block)

    model = Model(inputs=input_layer, outputs=block)

    print('learning rate', learning_rate)
    optimizer = RAdamOptimizer(learning_rate)
    optimizer = Adam(learning_rate)
    optimizer = runai.ga.keras.optimizers.Optimizer(optimizer, steps=gradient_accumulation)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

    model.name = name
    model.weight_filename = weight_filename
    model.class_weight = class_weight
    if load:
        print('load weights')
        model.load_weights(weight_filename)

    os.makedirs(os.path.split(summary_filename)[0], exist_ok=True)
    with open(summary_filename, 'w') as file:
        old = sys.stdout
        sys.stdout = file
        model.summary()
        sys.stdout = old
    return model
