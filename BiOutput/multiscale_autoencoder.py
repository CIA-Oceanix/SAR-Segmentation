import sys
import os
import numpy as np

from keras.optimizers import Adam
from keras.models import Model
from keras.layers import Conv2D, Conv2DTranspose, Lambda, MaxPooling2D
from keras.layers import Input, Dense, Dropout, Flatten, GlobalAveragePooling2D, concatenate, Reshape, UpSampling2D
from keras_radam.training import RAdamOptimizer

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


def build_encoder(input_shape, latent_space_shapes, activation, labels, conv_layers):
    input_layer = Input(input_shape)
    latent_space_layers = []
    block = None
    current_size = input_shape[0]
    final_size = input_shape[0] // (2 ** len(conv_layers))
    for i, (neurons, latent_space_shape) in enumerate(zip(conv_layers, latent_space_shapes)):
        if block is None:
            block, _ = convolution_block(input_layer, neurons, activation=activation)
        else:
            block, _ = convolution_block(block, neurons, activation=activation)

        # global_pooling = GlobalAveragePooling2D()(block)
        pooling_size = 2 ** (int(np.log2(current_size) - np.log2(final_size)) - 1)
        if pooling_size != 1:
            pooling = MaxPooling2D(pool_size=(pooling_size, pooling_size))(block)
        else:
            pooling = block
        global_pooling = Flatten()(pooling)
        dense_layer = Dense(np.prod(latent_space_shape), activation=activation)(global_pooling)
        if i == len(conv_layers) - 1:
            latent_space_layer = Dense(np.prod(latent_space_shapes[i]) - len(labels), activation=activation,
                                       name=f"latent_dense{i}")(dense_layer)
        else:
            latent_space_layer = Dense(np.prod(latent_space_shapes[i]), activation=activation,
                                       name=f"latent_dense{i}")(dense_layer)
        latent_space_layers.append(latent_space_layer)
        current_size = current_size // 2

    categorization_layer = Dense(len(labels), activation='softmax', name=f"categorization_layer")(global_pooling)
    latent_space_layers.append(categorization_layer)
    return Model(inputs=[input_layer], outputs=latent_space_layers, name="encoder")


def build_decoder(input_shape, latent_space_shapes, labels, conv_layers, activation):
    block = None
    input_layers = [Input([np.prod(latent_space_shape)]) for latent_space_shape in latent_space_shapes[:-1]]
    input_layers.append(Input([np.prod(latent_space_shapes[-1]) - len(labels)]))
    input_layers.append(Input([len(labels)]))

    concatenation = concatenate(input_layers[-2:])

    latent_space_layers = input_layers[:-2]
    latent_space_layers.append(concatenation)

    latent_space_layers = [Dense(np.prod(latent_space_shape), activation=activation)(layer)
                           for layer, latent_space_shape in zip(latent_space_layers, latent_space_shapes)]
    latent_space_layers = [Reshape(latent_space_shape)(layer)
                           for layer, latent_space_shape in zip(latent_space_layers, latent_space_shapes)]

    for i, (neurons, latent_space_layer) in enumerate(zip(conv_layers[::-1], latent_space_layers[::-1])):
        if block is not None:
            upsample = UpSampling2D((2 ** i, 2 ** i))(latent_space_layer)
            block = concatenate([upsample, block])
        else:
            block = latent_space_layer
        block, _ = convolution_block(block, neurons, activation=activation, maxpool=False)
        if i != len(conv_layers) - 1:
            block = Conv2DTranspose(neurons, strides=(2, 2), kernel_size=(4, 4),
                                    padding='same', activation=activation)(block)
    block = Conv2DTranspose(neurons, strides=(2, 2), kernel_size=(4, 4),
                            padding='same', activation=activation)(block)

    decoder_layer = Conv2D(input_shape[-1], (1, 1), activation='relu', name='decoder_layer')(block)

    return Model(inputs=input_layers, outputs=[decoder_layer], name="decoder")


def import_model(labels, weight_root=WEIGHT_ROOT, summary_root=SUMMARY_ROOT, load=LOAD,
                 learning_rate=LEARNING_RATE, name=DEFAULT_NAME, config=CONFIG):
    weight_filename = os.path.join(weight_root, f"{name}.h5")
    summary_filename = os.path.join(summary_root, f"{name}.txt")

    latent_space_shapes = config['LATENT_SPACE']
    conv_layers = config['CONV_LAYERS']
    input_shape = config.get('INPUT_SHAPE', (512, 512, 1))
    activation = config.get('ACTIVATION', 'relu')

    encoder = build_encoder(input_shape=input_shape, latent_space_shapes=latent_space_shapes,
                            activation=activation, labels=labels, conv_layers=conv_layers)
    decoder = build_decoder(input_shape=input_shape, latent_space_shapes=latent_space_shapes, labels=labels,
                            conv_layers=conv_layers, activation=activation)

    img = Input(shape=input_shape)
    encoded_repr = encoder(img)
    reconstructed_img = decoder(encoded_repr)

    model = Model(img, [encoded_repr[-1], reconstructed_img], name="total")
    model.compile(loss={'encoder': 'categorical_crossentropy', 'decoder': 'mae'},
                  loss_weights={"encoder": 0.1, "decoder": 1.0},
                  optimizer=RAdamOptimizer(learning_rate),
                  metrics={'classifier': 'acc'})

    model.name = name
    model.weight_filename = weight_filename
    if load:
        print('load weights')
        model.load_weights(weight_filename)

    with open(summary_filename, 'w') as file:
        old = sys.stdout
        sys.stdout = file
        print('Encoder')
        encoder.summary()
        print('\nDecoder')
        decoder.summary()
        print('\nTotal model')
        model.summary()
        sys.stdout = old
    return model
