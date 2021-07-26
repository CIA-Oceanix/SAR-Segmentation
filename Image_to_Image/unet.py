import os

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, RepeatVector, Reshape, concatenate, AveragePooling2D
from tensorflow.keras.optimizers import Adam
import tensorflow.keras.backend as K

from Rignak_Misc.path import get_local_file
from Rignak_DeepLearning.models import convolution_block, deconvolution_block, write_summary, load_weights
from Rignak_DeepLearning.config import get_config
from Rignak_DeepLearning.loss import LOSS_TRANSLATION, get_metrics

ROOT = get_local_file(__file__, os.path.join('..', '_outputs'))
LEARNING_RATE = 10 ** -5

CONFIG_KEY = 'segmenter'
CONFIG = get_config()[CONFIG_KEY]
DEFAULT_NAME = CONFIG.get('NAME', 'DEFAULT_MODEL_NAME')


def get_additional_metrics(metrics, loss, labels):
    if metrics is not None and 'all' in metrics:
        metrics.pop(metrics.index('all'))
        names = [f"{label}_loss" for label in labels]
        metrics += get_metrics(loss, names)
    return metrics


def build_encoder(input_shape, conv_layers, activation, batch_normalization, resnet):
    convs = []
    input_layer = Input(input_shape)
    block = input_layer

    for neurons in conv_layers:
        if resnet:
             downscaled_layer = AveragePooling2D(pool_size=(2, 2))(block)
             
        block, conv = convolution_block(block, neurons, activation=activation, maxpool=True,
                                        batch_normalization=batch_normalization)
        convs.append(conv)
        if resnet:
            block = concatenate([block, downscaled_layer], axis=-1)
            
    return input_layer, block, convs


def build_decoder(block, convs, conv_layers, output_shape, activation, last_activation, skip):
    for neurons, previous_conv in zip(conv_layers[::-1], convs[::-1]):
        if block.shape[1] == output_shape[0] and block.shape[2] == output_shape[1]:
            break
        previous_conv = previous_conv if skip else None
        block = deconvolution_block(block, previous_conv, neurons, activation=activation)

    output_layer = Conv2D(output_shape[-1], (1, 1), activation=last_activation)(block)
    return output_layer


def build_unet(input_shape, activation, batch_normalization, conv_layers, skip, last_activation, output_shape,
               central_shape, resnet, name):
    input_layer, block, convs = build_encoder(input_shape, conv_layers, activation, batch_normalization, resnet)

    block, conv = convolution_block(block, central_shape, activation=activation, maxpool=False,
                                    batch_normalization=batch_normalization)

    output_layer = build_decoder(block, convs, conv_layers, output_shape, activation, last_activation, skip)
    model = Model(inputs=[input_layer], outputs=[output_layer], name=name)
    return model


def build_multi_input_unet(input_shape, activation, batch_normalization, conv_layers, skip, last_activation,
                           output_shape, central_shape, additional_input_number, resnet, name):
    input_layer, block, convs = build_encoder(input_shape, conv_layers, activation, batch_normalization, resnet)

    block, conv = convolution_block(block, central_shape, activation=activation, maxpool=False,
                                    batch_normalization=batch_normalization)
    additional_inputs = Input(shape=(additional_input_number,))
    tiled_inputs = RepeatVector(int(block.shape[1] * block.shape[2]))(additional_inputs)
    tiled_inputs = Reshape([int(block.shape[1]), int(block.shape[2]), additional_input_number])(tiled_inputs)

    block = concatenate([block, tiled_inputs], axis=-1)

    output_layer = build_decoder(block, convs, conv_layers, output_shape, activation, last_activation, skip)
    model = Model(inputs=[input_layer, additional_inputs], outputs=[output_layer], name=name)
    return model


def import_model(root=ROOT, learning_rate=LEARNING_RATE,
                 config=CONFIG, name=DEFAULT_NAME, skip=True, metrics=None, labels=None, additional_input_number=0):
    load = config.get('LOAD', False)
    freeze = config.get('FREEZE', False)
    conv_layers = config['CONV_LAYERS']
    central_shape = config.get('CENTRAL_SHAPE', conv_layers[-1] * 2)
    learning_rate = config.get('LEARNING_RATE', learning_rate)
    batch_normalization = config.get('BATCH_NORMALIZATION', False)
    input_shape = config.get('INPUT_SHAPE', (512, 512, 3))
    output_shape = config.get('OUTPUT_SHAPE', input_shape)
    skip = config.get('SKIP', skip)
    resnet = config.get('RESNET', False)

    activation = config.get('ACTIVATION', 'relu')
    if activation == 'sin':
        activation = K.sin
    last_activation = config.get('LAST_ACTIVATION', 'sigmoid')
    if last_activation == 'sin':
        last_activation = K.sin

    loss = config.get('LOSS', 'mse')
    loss = LOSS_TRANSLATION.get(loss, loss)

    if labels is not None and len(labels) != output_shape[-1]:
        labels = [f'class_{i}' for i in range(output_shape[-1])]

    optimizer = Adam(learning_rate=learning_rate)
    metrics = get_additional_metrics(metrics, loss, labels)

    if additional_input_number:
        model = build_multi_input_unet(input_shape, activation, batch_normalization, conv_layers, skip, last_activation,
                                       output_shape, central_shape, additional_input_number, resnet, name)
    else:
        model = build_unet(input_shape, activation, batch_normalization, conv_layers, skip, last_activation,
                           output_shape, central_shape, resnet, name)
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics, run_eagerly=True)

    model.weight_filename = os.path.join(root, name, "model.h5")
    model.summary_filename = os.path.join(root, name, "model.txt")
        
    load_weights(model, model.weight_filename, load, freeze)
    write_summary(model)
    return model
