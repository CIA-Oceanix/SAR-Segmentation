import sys
import os

from keras.optimizers import Adam
from keras.models import Model, load_model
from keras.layers import Input, Conv2D, Dropout, concatenate

from Rignak_Misc.path import get_local_file
from Rignak_DeepLearning.models import convolution_block
from Rignak_DeepLearning.config import get_config
from Rignak_DeepLearning.loss import dice_coef_loss

WEIGHT_ROOT = get_local_file(__file__, os.path.join('..', '_outputs', 'models'))
SUMMARY_ROOT = get_local_file(__file__, os.path.join('..', '_outputs', 'summary'))
LOAD = False

LEARNING_RATE = 10 ** -4

CONFIG_KEY = 'saliency'
CONFIG = get_config()[CONFIG_KEY]
DEFAULT_NAME = CONFIG.get('NAME', 'DEFAULT_MODEL_NAME')

DROPOUT = 0.


def import_model(weight_root=WEIGHT_ROOT, summary_root=SUMMARY_ROOT, load=LOAD, learning_rate=LEARNING_RATE,
                 name=DEFAULT_NAME, config=CONFIG, dropout=DROPOUT):
    weight_filename = os.path.join(weight_root, f"{name}.h5")
    summary_filename = os.path.join(summary_root, f"{name}.txt")
    convs = []
    block = None

    conv_layers = config['CONV_LAYERS']
    input_shape = config.get('INPUT_SHAPE', (256, 256, 3))
    activation = config.get('ACTIVATION', 'relu')
    last_activation = config.get('ACTIVATION', 'sigmoid')
    output_canals = config.get('OUTPUT_CANALS', input_shape[-1])

    if 'saliency' in name:
        loss = 'binary_crossentropy'
    elif output_canals == 1:
        loss = dice_coef_loss
    else:
        loss = 'mse'

    pretrained_model_filename = os.path.join(weight_root, f"{name.replace('saliency', 'flat_autoencoder')}.h5")
    if 'saliency' in name:
        print('Will search for a pretrained encoder')
    if 'saliency' in name and os.path.exists(pretrained_model_filename):
        print('Found a flat_autoencoder model, will proceed to load it')
        model = load_model(pretrained_model_filename)
        # for layer in model.layers[:len(conv_layers) * 3 + 1]:
        #     layer.trainable = False
        inputs = model.input
        block = model.layers[-2].output
    else:
        inputs = Input(input_shape)
        # encoder
        for neurons in conv_layers:
            if block is None:
                block, conv = convolution_block(inputs, neurons, activation=activation, maxpool=False)
            else:
                block, conv = convolution_block(block, neurons, activation=activation, maxpool=False)
            convs.append(conv)

        # central
        block, conv = convolution_block(block, conv_layers[-1] * 2, activation=activation, maxpool=False)

        # decoder
        for neurons, previous_conv in zip(conv_layers[::-1], convs[::-1]):
            previous_conv = Dropout(dropout)(previous_conv)
            block = concatenate([block, previous_conv], axis=3)
            block, _ = convolution_block(block, neurons, activation=activation, maxpool=False)

    outputs = Conv2D(output_canals, (1, 1), activation=last_activation, name='output')(block)

    model = Model(inputs=[inputs], outputs=[outputs])
    model.compile(optimizer=Adam(lr=learning_rate), loss=loss)

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
