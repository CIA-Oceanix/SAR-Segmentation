import os

from keras.applications.inception_v3 import InceptionV3
from keras.applications.inception_resnet_v2 import InceptionResNetV2
from keras.layers import Dense, GlobalAveragePooling2D, Input, concatenate
from keras.models import Model
from keras_radam.training import RAdamOptimizer
from runai.ga import keras as runai
import keras.backend as K

from Rignak_Misc.path import get_local_file
from Rignak_DeepLearning.config import get_config
from Rignak_DeepLearning.models import write_summary
from Rignak_DeepLearning.loss import LOSS_TRANSLATION

ROOT = get_local_file(__file__, os.path.join('..', '_outputs'))

LOAD = False
LEARNING_RATE = 10 ** -4
DEFAULT_LOSS = 'categorical_crossentropy'
DEFAULT_METRICS = ['accuracy']

CONFIG_KEY = 'segmenter'
CONFIG = get_config()[CONFIG_KEY]
DEFAULT_NAME = CONFIG.get('NAME', 'DEFAULT_MODEL_NAME')


def build_inception_v3(input_shape, last_activation, labels, last_dense=False, resnet=False):
    input_layer = Input(shape=input_shape)
    img_conc = concatenate([input_layer, input_layer, input_layer]) if input_shape[-1] == 1 else input_layer

    get_model = InceptionResNetV2 if resnet else InceptionV3
    base_model = get_model(input_tensor=img_conc, classes=1, include_top=False, weights=None)

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    if last_dense:
        x = Dense(128, activation='relu')(x)

    if last_activation == 'sin':
        last_activation = K.sin
        
    x = Dense(len(labels), activation=last_activation)(x)
    model = Model(input_layer, outputs=x)
    return model


def build_multi_input_inception_v3(input_shape, last_activation, labels,
                                   additional_input_number, resnet=False):
    input_layer = Input(shape=input_shape)
    img_conc = concatenate([input_layer, input_layer, input_layer]) if input_shape[-1] == 1 else input_layer

    get_model = InceptionResNetV2 if resnet else InceptionV3
    base_model = get_model(input_tensor=img_conc, classes=1, include_top=False, weights=None)

    x = base_model.output
    x = GlobalAveragePooling2D()(x)

    additional_inputs = Input(shape=(additional_input_number,))
    x = concatenate([x, additional_inputs])

    x = Dense(128, activation='relu')(x)
    x = Dense(len(labels), activation=last_activation)(x)
    model = Model([input_layer, additional_inputs], outputs=x)
    return model


def import_model_v3(config=CONFIG, name=DEFAULT_NAME, root=ROOT, load=LOAD, additional_input_number=0, resnet=False):
    input_shape = config.get('INPUT_SHAPE')
    labels = config.get('LABELS')
    last_activation = config.get('LAST_ACTIVATION', 'softmax')
    learning_rate = config.get('LEARNING_RATE', LEARNING_RATE)

    loss = config.get('LOSS', DEFAULT_LOSS)
    loss = LOSS_TRANSLATION.get(loss, loss)

    metrics = config.get('METRICS', DEFAULT_METRICS)

    if additional_input_number:
        model = build_multi_input_inception_v3(input_shape, last_activation, labels,
                                               additional_input_number, resnet=resnet)
    else:
        model = build_inception_v3(input_shape, last_activation, labels, resnet=resnet)

    optimizer = RAdamOptimizer(learning_rate)
    # optimizer = runai.optimizers.Adam(lr=learning_rate, steps=4)
    
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
    model.labels = labels
    model.name = name
    model.weight_filename = os.path.join(root, name, "model.h5")
    model.summary_filename = os.path.join(root, name, "model.txt")

    if load:
        print('load weights')
        model.load_weights(model.weight_filename)

    write_summary(model)
    return model
