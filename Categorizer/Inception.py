import os
import sys

from keras.applications.inception_v3 import InceptionV3
from keras.layers import Dense, GlobalAveragePooling2D, Input, concatenate
from keras.models import Model
from keras.optimizers import Adam
from keras_radam.training import RAdamOptimizer
import runai.ga.keras

from Rignak_DeepLearning.Categorizer.flat import WEIGHT_ROOT, SUMMARY_ROOT

from Rignak_DeepLearning.loss import get_polarisation_metric

# nohup python3.6 train.py inceptionV3 chen --run_on_gpu=0 --IMAGENET=transfer --INPUT_SHAPE="(299,299,3)" > nohup_gpu0.out &
# nohup python3.6 train.py inceptionV3 chen --run_on_gpu=1 --INPUT_SHAPE="(299,299,3)" > nohup_gpu1.out &
# nohup python3.6 train.py inceptionV3 chen --run_on_gpu=2 --IMAGENET=transfer --INPUT_SHAPE="(512,512,3)" --NAME="bigger_" > nohup_gpu2.out &

LOAD = False
IMAGENET = False
DEFAULT_LOSS = 'categorical_crossentropy'
DEFAULT_METRICS = ['accuracy', get_polarisation_metric(4)]
DEFAULT_METRICS = ['accuracy']
LAST_ACTIVATION = 'softmax'
LEARNING_RATE = 10 ** -5
GRADIENT_ACCUMULATION = 8


def import_model_v3(input_shape, output_shape, name, weight_root=WEIGHT_ROOT, summary_root=SUMMARY_ROOT, load=LOAD,
                    imagenet=IMAGENET, loss=DEFAULT_LOSS, metrics=DEFAULT_METRICS, last_activation=LAST_ACTIVATION,
                    class_weight=None, learning_rate=LEARNING_RATE, gradient_accumulation=GRADIENT_ACCUMULATION):
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
    x = Dense(output_shape, activation=last_activation)(x)
    model = Model(img_input, outputs=x)

    if imagenet == "fine-tuning":
        for layer in model.layers[:-1]:
            layer.trainable = False

    optimizer = RAdamOptimizer(learning_rate)
    # optimizer = Adam(learning_rate)
    # optimizer = runai.ga.keras.optimizers.Optimizer(optimizer, steps=gradient_accumulation)
    
    if class_weight is not None:
        model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
    else:
        model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    model.name = name
    model.weight_filename = os.path.join(weight_root, f"{name}.h5")
    model.summary_filename = os.path.join(summary_root, f"{name}.txt")
    model.class_weight = class_weight

    if load:
        print('load weights')
        model.load_weights(model.weight_filename)

    os.makedirs(os.path.split(model.summary_filename)[0], exist_ok=True)
    with open(model.summary_filename, 'w') as file:
        old = sys.stdout
        sys.stdout = file
        model.summary()
        sys.stdout = old
    return model
