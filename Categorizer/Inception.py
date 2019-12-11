import os
import sys

from keras.applications.inception_v3 import InceptionV3
from keras.layers import Dense, GlobalAveragePooling2D
from keras.models import Model

from Rignak_DeepLearning.Categorizer.flat import WEIGHT_ROOT, SUMMARY_ROOT

LOAD = False
IMAGENET = False


def import_model_v3(input_shape, output_shape, name, weight_root=WEIGHT_ROOT, summary_root=SUMMARY_ROOT, load=LOAD,
                    imagenet=IMAGENET):
    if imagenet:
        weights = "imagenet"
    else:
        weights = None

    base_model = InceptionV3(weights=weights, input_shape=input_shape, classes=output_shape, include_top=False)
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(output_shape, activation='softmax')(x)
    model = Model(base_model.input, outputs=x)

    model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

    if imagenet == "fine-tuning":
        for layer in base_model.layers:
            layer.trainable = False

    model.name = f"{name}_{load}"
    model.weight_filename = os.path.join(weight_root, f"{model.name}.h5")
    model.summary_filename = os.path.join(summary_root, f"{model.name}.txt")

    if load:
        print('load weights')
        model.load_weights(model.weight_filename)

    with open(model.summary_filename, 'w') as file:
        old = sys.stdout
        sys.stdout = file
        model.summary()
        sys.stdout = old
    return model
