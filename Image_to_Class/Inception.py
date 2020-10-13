import os

from keras.applications.inception_v3 import InceptionV3
from keras.layers import Dense, GlobalAveragePooling2D, Input, concatenate
from keras.models import Model
from keras_radam.training import RAdamOptimizer
from keras.metrics import categorical_accuracy, mean_squared_error, categorical_crossentropy

from Rignak_Misc.path import get_local_file
from Rignak_DeepLearning.loss import get_metrics
from Rignak_DeepLearning.models import write_summary

WEIGHT_ROOT = get_local_file(__file__, os.path.join('..', '_outputs', 'models'))
SUMMARY_ROOT = get_local_file(__file__, os.path.join('..', '_outputs', 'summary'))

LOAD = False
IMAGENET = False
DEFAULT_LOSS = 'categorical_crossentropy'
DEFAULT_METRICS = ['accuracy']
LAST_ACTIVATION = 'softmax'
LEARNING_RATE = 10 ** -5
GRADIENT_ACCUMULATION = 8


def import_model_v3(input_shape, output_shape, name, weight_root=WEIGHT_ROOT, summary_root=SUMMARY_ROOT, load=LOAD,
                    imagenet=IMAGENET, loss=DEFAULT_LOSS, metrics=DEFAULT_METRICS, last_activation=LAST_ACTIVATION,
                    learning_rate=LEARNING_RATE, last_dense=False):
    if imagenet:
        print('Will load imagenet weights')
        weights = "imagenet"
    else:
        weights = None

    if input_shape[-1] == 1:
        img_input = Input(shape=input_shape)
        img_conc = concatenate([img_input, img_input, img_input])
        base_model = InceptionV3(input_tensor=img_conc, classes=1, include_top=False)
    else:
        base_model = InceptionV3(weights=weights, input_shape=input_shape, classes=output_shape, include_top=False)
        img_input = base_model.input

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    if last_dense:
        x = Dense(128, activation='relu')(x)
    x = Dense(output_shape, activation=last_activation)(x)
    model = Model(img_input, outputs=x)

    if imagenet == "fine-tuning":
        for layer in model.layers[:-1]:
            layer.trainable = False

    optimizer = RAdamOptimizer(learning_rate)

    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    model.name = name
    model.weight_filename = os.path.join(weight_root, f"{name}.h5")
    model.summary_filename = os.path.join(summary_root, f"{name}.txt")

    if load:
        print('load weights')
        model.load_weights(model.weight_filename)

    write_summary(model)
    return model
