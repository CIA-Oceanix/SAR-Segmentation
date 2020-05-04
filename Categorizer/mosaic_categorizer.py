import os
import sys
import tqdm

# os.environ['CUDA_VISIBLE_DEVICES'] = "-1"

from Rignak_DeepLearning import deprecation_warnings

deprecation_warnings.filter_warnings()

from keras.applications.inception_v3 import InceptionV3
from keras.layers import Dense, GlobalAveragePooling2D, Input, concatenate, Lambda
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


def import_model(input_shape, output_shape, name, weight_root=WEIGHT_ROOT, summary_root=SUMMARY_ROOT, load=LOAD,
                 loss=DEFAULT_LOSS, metrics=DEFAULT_METRICS, last_activation=LAST_ACTIVATION,
                 class_weight=None, learning_rate=LEARNING_RATE):
    assert input_shape == (512, 512, 1)

    img_input = Input(shape=input_shape)

    inception_partial_model = InceptionV3(include_top=False, input_shape=(128, 128, 3))
    inception_model_output = inception_partial_model.output
    inception_model_output = GlobalAveragePooling2D()(inception_model_output)
    inception_model_output = Dense(output_shape, activation=last_activation)(inception_model_output)

    inception_model = Model(inception_partial_model.input, inception_model_output)

    def get_slice(i, j):
        def slice(x):
            return K.slice(x, [0, i * 128, j * 128, 0], [-1, 128, 128, -1])

        return slice

    layers = []
    for i in range(4):
        for j in range(4):
            layer = Lambda(get_slice(i, j))(img_input)
            layer = concatenate([layer, layer, layer])
            layer = Lambda(lambda x: K.expand_dims(x, axis=1))(layer)
            layers.append(layer)

    mosaic = concatenate(layers, axis=1)
    mosaic = Lambda(lambda x: K.reshape(x, (K.shape(x)[0] * K.shape(x)[1], 128, 128, 3)))(mosaic)
    mosaic = inception_model(mosaic)
    mosaic = Lambda(lambda x: K.reshape(x, (K.shape(x)[0] // 16, 16, output_shape)))(mosaic)
    mosaic = Lambda(lambda x: K.mean(x, axis=1))(mosaic)

    model = Model(img_input, outputs=mosaic)

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
    import_model((512, 512, 1), 10, 'default')