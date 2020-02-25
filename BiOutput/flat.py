import sys
import os

from keras.optimizers import Adam
from keras.models import Model
from keras.layers import Conv2D
from keras.layers import Input, Dense, Dropout, Flatten

from Rignak_Misc.path import get_local_file
from Rignak_DeepLearning.models import convolution_block, deconvolution_block
from Rignak_DeepLearning.config import get_config

WEIGHT_ROOT = get_local_file(__file__, os.path.join('..', '_outputs', 'models'))
SUMMARY_ROOT = get_local_file(__file__, os.path.join('..', '_outputs', 'summary'))
LOAD = False

LEARNING_RATE = 10 ** -4

CONFIG_KEY = 'bimode'
CONFIG = get_config()[CONFIG_KEY]
DEFAULT_NAME = CONFIG.get('NAME', 'DEFAULT_MODEL_NAME')


def import_model(output_canals, labels, weight_root=WEIGHT_ROOT, summary_root=SUMMARY_ROOT, load=LOAD,
                 learning_rate=LEARNING_RATE, name=DEFAULT_NAME, config=CONFIG):
    weight_filename = os.path.join(weight_root, f"{name}.h5")
    summary_filename = os.path.join(summary_root, f"{name}.txt")

    conv_layers = config['CONV_LAYERS']
    dense_layers = config['DENSE_LAYERS']
    input_shape = config.get('INPUT_SHAPE', (256, 256, 3))
    activation = config.get('ACTIVATION', 'relu')

    input_layer = Input(input_shape)
    block = None
    for neurons in conv_layers:
        if block is None:
            block, _ = convolution_block(input_layer, neurons, activation=activation)
        else:
            block, _ = convolution_block(block, neurons, activation=activation)

    categorizer_block = Flatten()(block)
    for neurons in dense_layers:
        categorizer_block = Dense(neurons, activation=activation)( categorizer_block)
        categorizer_block = Dropout(0.5)(categorizer_block)
    categorizer_block = Dense(len(labels), activation='softmax', name='categorizer_block')(categorizer_block)

    decoder_block = block
    for neurons in conv_layers[::-1]:
        decoder_block = deconvolution_block(decoder_block, None, neurons, activation=activation)
    decoder_block = Conv2D(output_canals, (1, 1), activation='sigmoid', name='decoder_block')(decoder_block)

    model = Model(inputs=input_layer, outputs=[categorizer_block, decoder_block])

    losses = {"categorizer_block": "categorical_crossentropy", "decoder_block": "mse"}
    loss_weights = {"categorizer_block": 1.0, "decoder_block": 1.}
    model.compile(optimizer=Adam(lr=learning_rate), loss=losses, loss_weights=loss_weights,
                  metrics={'categorizer_block': 'accuracy'})

    model.name = name
    model.weight_filename = weight_filename
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
