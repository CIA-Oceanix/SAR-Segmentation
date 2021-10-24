import os

import tensorflow.keras.backend as K
from tensorflow.keras.layers import Input, concatenate, AveragePooling2D, GlobalAveragePooling2D, \
    Dense, Lambda
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.optimizers import Adam

from Rignak_DeepLearning.config import get_config
from Rignak_DeepLearning.models import convolution_block, write_summary
from Rignak_Misc.path import get_local_file

ROOT = get_local_file(__file__, os.path.join('..', '_outputs'))
LOAD = False
LEARNING_RATE = 10 ** -5

CONFIG_KEY = 'segmenter'
CONFIG = get_config()[CONFIG_KEY]
DEFAULT_NAME = CONFIG.get('NAME', 'DEFAULT_MODEL_NAME')


def build_distance_model(input_shape, model_path, layer_name, model_name):
    input_layer_1 = Input(input_shape)
    input_layer_2 = Input(input_shape)

    model = load_model(model_path)
    layer = model.get_layer(layer_name)
    encoder_model = Model(inputs=model.inputs, outputs=[layer.output], name="encoder")
    #for layer in encoder_model.layers:
    #   layer.trainable = False

    intermediate_layer_1 = encoder_model(input_layer_1)
    intermediate_layer_2 = encoder_model(input_layer_2)
    diff_layer = Lambda(lambda x: K.abs(x[0]-x[1]), name="diff_layer")([intermediate_layer_1, intermediate_layer_2])
    #intermediate_layer = concatenate([intermediate_layer_1, intermediate_layer_2, diff_layer])
    #output_layer = Dense(output_layer.shape[-1].value, activation="relu")(output_layer)
    output_layer = Dense(1, activation="sigmoid")(diff_layer)

    model = Model(inputs=[input_layer_1, input_layer_2], outputs=output_layer, name=model_name)
    return model

from tensorflow.keras.losses import binary_crossentropy, mean_squared_error
from tensorflow.keras.metrics import binary_accuracy


def siamese_loss(y_true, y_pred):
    return mean_squared_error(y_true[:, 2], y_pred[:, 0])


def siamese_acc(y_true, y_pred):
    return K.greater(0.5, K.abs(y_pred[:, 0] - y_true[:, 2]))


def import_model(root=ROOT, load=LOAD, learning_rate=LEARNING_RATE, config=CONFIG, name=DEFAULT_NAME):
    learning_rate = config.get('LEARNING_RATE', learning_rate)
    input_shape = config.get('INPUT_SHAPE', (512, 512, 3))

    optimizer = Adam(learning_rate)

    metric = siamese_acc
    metric.name = "accuracy"

    model = build_distance_model(input_shape, config['MODEL_PATH'], config['LAYER_NAME'], name)
    model.compile(optimizer=optimizer, loss=siamese_loss, metrics=[metric])

    model.__name = name
    model.weight_filename = os.path.join(root, name, "model.h5")
    model.summary_filename = os.path.join(root, name, "model.txt")

    if load:
        print('load weights')
        model.load_weights(model.weight_filename)

    write_summary(model)
    return model
