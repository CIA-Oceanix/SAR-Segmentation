import sys
import os

from keras.optimizers import Adam
from keras.models import Model
from keras.layers import Conv2D
from keras.layers import Input, Dense, Dropout, Flatten, GlobalAveragePooling2D, concatenate, Reshape, UpSampling2D

from Rignak_Misc.path import get_local_file
from Rignak_DeepLearning.models import convolution_block
from Rignak_DeepLearning.config import get_config

WEIGHT_ROOT = get_local_file(__file__, os.path.join('..', '_outputs', 'models'))
SUMMARY_ROOT = get_local_file(__file__, os.path.join('..', '_outputs', 'summary'))
LOAD = False

LEARNING_RATE = 10 ** -4

CONFIG_KEY = 'multiscale_bimode'
CONFIG = get_config()[CONFIG_KEY]
DEFAULT_NAME = CONFIG.get('NAME', 'DEFAULT_MODEL_NAME')


def import_model(output_canals, labels, weight_root=WEIGHT_ROOT, summary_root=SUMMARY_ROOT, load=LOAD,
                 learning_rate=LEARNING_RATE, name=DEFAULT_NAME, config=CONFIG):
    weight_filename = os.path.join(weight_root, f"{name}.h5")
    summary_filename = os.path.join(summary_root, f"{name}.txt")

    latent_space_root_length = config['LATENT_SPACE']
    conv_layers = config['CONV_LAYERS']
    input_shape = config.get('INPUT_SHAPE', (512, 512, 1))
    activation = config.get('ACTIVATION', 'relu')

    print('INPUT_SHAPE', input_shape)

    input_layer = Input(input_shape)
    latent_space_layers = []
    block = None
    for i, neurons in enumerate(conv_layers):
        if block is None:
            block, _ = convolution_block(input_layer, neurons, activation=activation)
        else:
            block, _ = convolution_block(block, neurons, activation=activation)

        global_pooling = GlobalAveragePooling2D()(block)
        if i == len(conv_layers) - 1:
            latent_space_layer = Dense(latent_space_root_length ** 2 - len(labels), activation=activation,
                                       name=f"latent_dense{i}")(global_pooling)
        else:
            latent_space_layer = Dense(latent_space_root_length ** 2, activation=activation,
                                       name=f"latent_dense{i}")(global_pooling)
        latent_space_layers.append(latent_space_layer)

    categorization_layer = Dense(len(labels), activation='softmax', name=f"categorization_layer")(global_pooling)
    latent_space_layers[-1] = concatenate([categorization_layer, latent_space_layers[-1]])

    latent_space_layers = [Reshape((latent_space_root_length, latent_space_root_length, 1))(layer)
                           for layer in latent_space_layers]

    block = None
    for i, (neurons, latent_space_layer) in enumerate(zip(conv_layers[::-1], latent_space_layers[::-1])):
        if block is not None:
            upsample = UpSampling2D((2 ** i, 2 ** i))(latent_space_layer)
            block = concatenate([upsample, block])
        else:
            block = latent_space_layer
        block, _ = convolution_block(block, neurons, activation=activation, maxpool=False)
        if i != len(conv_layers)-1:
            block = UpSampling2D((2, 2))(block)

    decoder_layer = Conv2D(output_canals, (1, 1), activation='sigmoid', name='decoder_layer')(block)

    model = Model(inputs=input_layer, outputs=[categorization_layer, decoder_layer])

    losses = {"categorization_layer": "categorical_crossentropy", "decoder_layer": "mse"}
    loss_weights = {"categorization_layer": 1., "decoder_layer": 1.}
    model.compile(optimizer=Adam(lr=learning_rate), loss=losses, loss_weights=loss_weights,
                  metrics={'categorization_layer': 'accuracy'})

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
