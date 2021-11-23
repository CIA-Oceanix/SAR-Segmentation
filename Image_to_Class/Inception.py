import os

from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.applications.inception_resnet_v2 import InceptionResNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Input, concatenate
from tensorflow.keras.models import Model, load_model

#from keras_radam.optimizers import RAdam
from tensorflow.keras.optimizers import Adam
#from tensorflow_addons.optimizers import rectified_adam as RAdam

import tensorflow.keras.backend as K

from Rignak_Misc.path import get_local_file
from Rignak_DeepLearning.config import get_config
from Rignak_DeepLearning.models import write_summary, load_weights, flatten_model
from Rignak_DeepLearning.loss import LOSS_TRANSLATION

ROOT = get_local_file(__file__, os.path.join('..', '_outputs'))

LEARNING_RATE = 10 ** -4
DEFAULT_LOSS = 'categorical_crossentropy'
DEFAULT_METRICS = ['accuracy']

CONFIG_KEY = 'segmenter'
CONFIG = get_config()[CONFIG_KEY]
DEFAULT_NAME = CONFIG.get('NAME', 'DEFAULT_MODEL_NAME')


def build_inception_v3(input_shape, last_activation, labels, name, last_dense=False, resnet=False):
    input_layer = Input(shape=input_shape)
    img_conc = concatenate([input_layer, input_layer, input_layer]) if input_shape[-1] == 1 else input_layer

    get_model = InceptionResNetV2 if resnet else InceptionV3
    base_model = get_model(input_tensor=img_conc, classes=1, include_top=False, weights='imagenet')

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    if last_dense:
        x = Dense(128, activation='relu')(x)
        
    if last_activation == 'sin':
        last_activation = K.sin
        
    x = Dense(len(labels), activation=last_activation)(x)
    model = Model(input_layer, outputs=x, name=name)
    return model


def build_multi_input_inception_v3(input_shape, last_activation, labels, name,
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
    x = Dense(128, activation='relu')(x)
    x = Dense(len(labels), activation=last_activation)(x)
    model = Model([input_layer, additional_inputs], outputs=x, name=name)
    return model


def import_model_v3(config=CONFIG, name=DEFAULT_NAME, root=ROOT, additional_input_number=0, resnet=False):
    load = config.get('LOAD', False)
    freeze = config.get('FREEZE', False)
    input_shape = list(config.get('INPUT_SHAPE'))
    
    flatten_axis = None
    if input_shape[1] == 1: 
        flatten_axis = 1
        input_shape[1] = input_shape[0]
    if input_shape[0] == 1: 
        flatten_axis = 0
        input_shape[0] = input_shape[1]
    
    labels = config.get('LABELS')
    last_activation = config.get('LAST_ACTIVATION', 'softmax')
    learning_rate = config.get('LEARNING_RATE', LEARNING_RATE)
    last_dense=config.get('LAST_DENSE', False)

    loss = config.get('LOSS', DEFAULT_LOSS)
    loss = LOSS_TRANSLATION.get(loss, loss)

    metrics = config.get('METRICS', DEFAULT_METRICS)

    print('additional_input_number', last_dense)
    if additional_input_number:
        model = build_multi_input_inception_v3(input_shape, last_activation, labels, name,
                                               additional_input_number, resnet=resnet)
    else:
        model = build_inception_v3(input_shape, last_activation, labels, name, resnet=resnet, 
                                   last_dense=last_dense)
    if flatten_axis is not None:
        model = flatten_model(model, axis=flatten_axis)

    #optimizer = RAdam(learning_rate)
    optimizer = Adam(learning_rate)
    #optimizer = runai.optimizers.Adam(lr=learning_rate, steps=4)
    
    model.labels = labels
    model.weight_filename = os.path.join(root, name, "model.h5")
    model.summary_filename = os.path.join(root, name, "model.txt")

    load_weights(model, model.weight_filename, load, freeze)
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
    write_summary(model)
    return model
