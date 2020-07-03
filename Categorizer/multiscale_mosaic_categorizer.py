import os
import sys

# os.environ['CUDA_VISIBLE_DEVICES'] = "-1"

from Rignak_DeepLearning import deprecation_warnings

deprecation_warnings.filter_warnings()

from keras.applications.inception_v3 import InceptionV3
from keras.layers import Dense, GlobalAveragePooling2D, Input, concatenate, Lambda, add, Flatten, average
from keras.models import Model
from keras_radam.training import RAdamOptimizer
import keras.backend as K

from Rignak_DeepLearning.Categorizer.flat import WEIGHT_ROOT, SUMMARY_ROOT

LOAD = False
IMAGENET = False
DEFAULT_LOSS = 'categorical_crossentropy'
DEFAULT_METRICS = ['accuracy']
LAST_ACTIVATION = 'softmax'
LEARNING_RATE = 10 ** -5
MODALITY = 'mean'


def import_model(input_shape, output_shape, name, weight_root=WEIGHT_ROOT, summary_root=SUMMARY_ROOT, load=LOAD,
                 loss=DEFAULT_LOSS, metrics=DEFAULT_METRICS, last_activation=LAST_ACTIVATION,
                 class_weight=None, learning_rate=LEARNING_RATE, modality=MODALITY):
    assert input_shape == (512, 512, 1)

    img_input = Input(shape=input_shape)

    inception_partial_model = InceptionV3(include_top=False)
    inception_partial_model = Model(inception_partial_model.layers[1].input, inception_partial_model.output)

    inception_model_output = inception_partial_model.output
    inception_model_output = GlobalAveragePooling2D()(inception_model_output)
    inception_model_output = Dense(output_shape, activation=last_activation)(inception_model_output)

    inception_model = Model(inception_partial_model.input, inception_model_output)

    def get_slice(i, j, size=128):
        def slice(x):
            return K.slice(x, [0, i * size // 2, j * size // 2, 0], [-1, size, size, -1])

        return slice

    def get_layers(i_max, j_max, size):
        layers = []
        for i in range(i_max):
            for j in range(j_max):
                layer = Lambda(get_slice(i, j, size=size))(img_input)
                layer = concatenate([layer, layer, layer])
                layer = Lambda(lambda x: K.expand_dims(x, axis=1))(layer)
                layers.append(layer)
        return layers

    layers128 = get_layers(7, 7, 128)
    layers256 = get_layers(3, 3, 256)
    layers512 = get_layers(1, 1, 512)

    def get_mosaic_output(layers, n_tiles, size):
        if len(layers) > 1:
            mosaic = concatenate(layers, axis=1)
        else:
            mosaic = layers[0]
        mosaic = Lambda(lambda x: K.reshape(x, (K.shape(x)[0] * K.shape(x)[1], size, size, 3)))(mosaic)
        mosaic = inception_model(mosaic)
        mosaic = Lambda(lambda x: K.reshape(x, (K.shape(x)[0] // n_tiles, n_tiles, output_shape)),
                        name=f"mosaic{size}")(mosaic)

        if modality == "auto":
            output = Flatten()(mosaic)

        if 'mean' in modality:
            mean_mosaic = Lambda(lambda x: K.mean(x, axis=1))(mosaic)
            output = mean_mosaic
        if 'max' in modality:
            max_mosaic = Lambda(lambda x: K.max(x, axis=1))(mosaic)
            output = max_mosaic
        if 'min' in modality:
            min_mosaic = Lambda(lambda x: K.min(x, axis=1))(mosaic)
            output = max_mosaic

        if modality == 'mean_and_max':
            mean_and_max_mosaic = add([max_mosaic, mean_mosaic])
            output = mean_and_max_mosaic
        elif modality == 'mean_or_max':
            local_mean_mosaic = Lambda(lambda x: x * K.constant([[1, 0, 1, 0, 0, 1, 0, 1, 0, 0]]))(mean_mosaic)
            global_max_mosaic = Lambda(lambda x: x * K.constant([[0, 1, 0, 1, 1, 0, 1, 0, 1, 1]]))(max_mosaic)
            output = add([global_max_mosaic, local_mean_mosaic])
        elif modality == 'mean_or_max_inv':
            local_max_mosaic = Lambda(lambda x: x * K.constant([[1, 0, 1, 0, 0, 1, 0, 1, 0, 0]]))(max_mosaic)
            global_mean_mosaic = Lambda(lambda x: x * K.constant([[0, 1, 0, 1, 1, 0, 1, 0, 1, 1]]))(mean_mosaic)
            output = add([local_max_mosaic, global_mean_mosaic])
        elif modality == 'min_or_max':
            local_min_mosaic = Lambda(lambda x: x * K.constant([[1, 0, 1, 0, 0, 1, 0, 1, 0, 0]]))(min_mosaic)
            global_max_mosaic = Lambda(lambda x: x * K.constant([[0, 1, 0, 1, 1, 0, 1, 0, 1, 1]]))(max_mosaic)
            output = add([global_max_mosaic, local_min_mosaic])
        elif modality == 'min_or_max_inv':
            local_max_mosaic = Lambda(lambda x: x * K.constant([[1, 0, 1, 0, 0, 1, 0, 1, 0, 0]]))(max_mosaic)
            global_min_mosaic = Lambda(lambda x: x * K.constant([[0, 1, 0, 1, 1, 0, 1, 0, 1, 1]]))(min_mosaic)
            output = add([local_max_mosaic, global_min_mosaic])
        return output

    output128 = get_mosaic_output(layers128, 49, 128)
    output256 = get_mosaic_output(layers256, 9, 256)
    output512 = get_mosaic_output(layers512, 1, 512)

    if modality == 'auto':
        output = concatenate([output128, output256, output512])
        output = Dense(10, activation=last_activation)(output)
    else:
        output = average([output128, output256, output512])

    model = Model(img_input, outputs=output)

    optimizer = RAdamOptimizer(learning_rate)
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


if __name__ == '__main__':
    model = import_model((512, 512, 1), 10, 'default', modality='min_or_max_inv')
