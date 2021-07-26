import os

from keras.models import Model
from keras.layers import Input, Conv2D, RepeatVector, Reshape, concatenate, AveragePooling2D, GlobalAveragePooling2D, \
    Dense
from keras_radam.training import RAdamOptimizer
import keras.backend as K

from Rignak_Misc.path import get_local_file
from Rignak_DeepLearning.models import convolution_block, deconvolution_block, write_summary
from Rignak_DeepLearning.config import get_config
from Rignak_DeepLearning.loss import LOSS_TRANSLATION

from Rignak_DeepLearning.Image_to_Distance.layers import CorrLayer

ROOT = get_local_file(__file__, os.path.join('..', '_outputs'))
LOAD = False
LEARNING_RATE = 10 ** -5

CONFIG_KEY = 'segmenter'
CONFIG = get_config()[CONFIG_KEY]
DEFAULT_NAME = CONFIG.get('NAME', 'DEFAULT_MODEL_NAME')


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

    return Model(inputs=input_layer, outputs=convs, name="encoder")


def build_correlation_block(submodel_outputs_1, submodel_outputs_2, resnet, activation, batch_normalization):
    previous_layer = None
    i_skip = 2
    for i, (submodel_output_1, submodel_output_2) in enumerate(zip(submodel_outputs_1, submodel_outputs_2)):
        if i < i_skip: continue
        current_layer = CorrLayer()([submodel_output_1, submodel_output_2])

        if previous_layer is not None:
            current_layer = concatenate([current_layer, previous_layer], axis=-1)
        previous_layer, _ = convolution_block(
            current_layer, submodel_output_1.shape[-1].value,
            activation=activation, maxpool=True,
            batch_normalization=batch_normalization, block_depth=2
        )
    return previous_layer


def build_distance_model(input_shape, activation, batch_normalization, conv_layers, resnet):
    input_layer_1 = Input(input_shape)
    input_layer_2 = Input(input_shape)

    model = build_encoder(input_shape, conv_layers, activation="relu", batch_normalization=False, resnet=True)

    output_layer = build_correlation_block(
        model(input_layer_1), model(input_layer_2),
        resnet=False, activation="relu", batch_normalization=False
    )
    output_layer = GlobalAveragePooling2D()(output_layer)
    output_layer = Dense(1, activation="sigmoid")(output_layer)
    model = Model(inputs=[input_layer_1, input_layer_2], outputs=output_layer)
    return model


from keras.losses import mean_squared_error
from keras.metrics import CategoricalAccuracy


def siamese_loss(y_true, y_pred):
    return mean_squared_error(y_true[:, 2], y_pred)


def siamese_acc(y_true, y_pred):
    return CategoricalAccuracy(y_true[:, 2], y_pred)


def import_model(root=ROOT, load=LOAD, learning_rate=LEARNING_RATE, config=CONFIG, name=DEFAULT_NAME):
    conv_layers = config['CONV_LAYERS']
    learning_rate = config.get('LEARNING_RATE', learning_rate)
    batch_normalization = config.get('BATCH_NORMALIZATION', False)
    input_shape = config.get('INPUT_SHAPE', (512, 512, 3))
    resnet = config.get('RESNET', False)

    activation = config.get('ACTIVATION', 'relu')
    if activation == 'sin': activation = K.sin

    loss = config.get('LOSS', 'mse')
    loss = LOSS_TRANSLATION.get(loss, loss)

    optimizer = RAdamOptimizer(learning_rate)

    model = build_distance_model(input_shape, activation, batch_normalization, conv_layers, resnet)
    model.compile(optimizer=optimizer, loss=siamese_loss)

    model.name = name
    model.weight_filename = os.path.join(root, name, "model.h5")
    model.summary_filename = os.path.join(root, name, "model.txt")

    if load:
        print('load weights')
        model.load_weights(model.weight_filename)

    write_summary(model)
    return model
