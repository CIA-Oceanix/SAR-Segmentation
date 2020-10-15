import os

from keras.applications.inception_v3 import InceptionV3
from keras.layers import Dense, GlobalAveragePooling2D, Input, concatenate
from keras.models import Model
from keras_radam.training import RAdamOptimizer

from Rignak_Misc.path import get_local_file
from Rignak_DeepLearning.config import get_config
from Rignak_DeepLearning.models import write_summary

WEIGHT_ROOT = get_local_file(__file__, os.path.join('..', '_outputs', 'models'))
SUMMARY_ROOT = get_local_file(__file__, os.path.join('..', '_outputs', 'summary'))

LOAD = False
LEARNING_RATE = 10 ** -5
DEFAULT_LOSS = 'categorical_crossentropy'
DEFAULT_METRICS = ['accuracy']

CONFIG_KEY = 'segmenter'
CONFIG = get_config()[CONFIG_KEY]
DEFAULT_NAME = CONFIG.get('NAME', 'DEFAULT_MODEL_NAME')


def import_model_v3(config=CONFIG, name=DEFAULT_NAME, weight_root=WEIGHT_ROOT, summary_root=SUMMARY_ROOT, load=LOAD,
                    last_dense=False):
    input_shape = config.get('INPUT_SHAPE')
    labels = config.get('LABELS')
    activation = config.get('ACTIVATION', 'relu')
    last_activation = config.get('LAST_ACTIVATION', 'softmax')
    learning_rate = config.get('LEARNING_RATE', LEARNING_RATE)

    imagenet = config.get('IMAGENET', False)
    loss = config.get('LOSS', DEFAULT_LOSS)
    metrics = config.get('METRICS', DEFAULT_METRICS)

    if imagenet:
        print('Will load imagenet weights')
        weights = "imagenet"
    else:
        weights = None

    if input_shape[-1] == 1:
        img_input = Input(shape=input_shape)
        img_conc = concatenate([img_input, img_input, img_input])
        base_model = InceptionV3(input_tensor=img_conc, classes=1, include_top=False, activation=activation)
    else:
        base_model = InceptionV3(weights=weights, input_shape=input_shape, classes=len(labels), include_top=False,
                                 activation=activation)
        img_input = base_model.input

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    if last_dense:
        x = Dense(128, activation='relu')(x)
    x = Dense(len(labels), activation=last_activation)(x)
    model = Model(img_input, outputs=x)

    if imagenet == "fine-tuning":
        for layer in model.layers[:-1]:
            layer.trainable = False

    optimizer = RAdamOptimizer(learning_rate)

    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
    model.labels = labels
    model.name = name
    model.weight_filename = os.path.join(weight_root, f"{name}.h5")
    model.summary_filename = os.path.join(summary_root, f"{name}.txt")

    if load:
        print('load weights')
        model.load_weights(model.weight_filename)

    write_summary(model)
    return model
