import os
import sys
import tqdm

# os.environ['CUDA_VISIBLE_DEVICES'] = "-1"

from Rignak_DeepLearning import deprecation_warnings

deprecation_warnings.filter_warnings()

from keras.applications.inception_v3 import InceptionV3
from keras.layers import Dense, GlobalAveragePooling2D, Input, concatenate, Lambda, average, Softmax, \
    multiply, add, ActivityRegularization
from keras.models import Model, load_model
from keras_radam.training import RAdamOptimizer
import keras.backend as K
from keras.losses import mse

from Rignak_DeepLearning.Categorizer.flat import WEIGHT_ROOT, SUMMARY_ROOT
from Rignak_DeepLearning.Autoencoders.unet import import_model as import_unet_model

LOAD = False
IMAGENET = False
DEFAULT_LOSS = 'mse'
DEFAULT_METRICS = ['accuracy']
LAST_ACTIVATION = 'softmax'
LEARNING_RATE = 10 ** -5


def import_model(input_shape, output_shape, name, config, load=LOAD, weight_root=WEIGHT_ROOT, summary_root=SUMMARY_ROOT,
                 class_weight=None):
    assert input_shape == (256, 256, 1)
    assert output_shape == 2

    last_activation = config.get('ACTIVATION', 'sigmoid')
    learning_rate = config.get('LEARNING_RATE', LEARNING_RATE)

    regularization_factor = config.get('REGULARIZATION_FACTOR', 1 / 256 / 256 / 32)

    unet_config = config.copy()
    unet_config['ACTIVATION'] = 'sigmoid'

    img_input = Input(shape=input_shape)
    # unet = import_unet_model(config=unet_config)
    unet = load_model("_outputs/models/TenGeoP-SARwv_heatmap/Iceberg256.h5")
    for layer in unet.layers:
        layer.trainable = False
    unet.name = 'unet'

    inception_partial_model = InceptionV3(include_top=False, input_shape=(input_shape[0], input_shape[1], 3))

    unet_output = unet(img_input)

    # binarization_layer = ActivityRegularization(l2=regularization_factor)(unet_output)
    binarization_layer = Lambda(lambda x: x / K.mean(x) / 1.5,
                                name="division")(unet_output)
    binarization_layer = Lambda(lambda x: x / 1,
                                name="division")(unet_output)
    binarization_layer = Lambda(lambda x: K.round(x), name='binarization')(binarization_layer)

    inverse_binarization_layer = Lambda(lambda x: K.abs(x - 1))(binarization_layer)

    random_layer = Lambda(lambda x: K.random_uniform(K.shape(x), 0, 1))(inverse_binarization_layer)
    random_layer = multiply([inverse_binarization_layer, random_layer])

    masked_input = multiply([img_input, binarization_layer])
    masked_input = add([masked_input, random_layer], name='masked_input')

    inception_input = concatenate([masked_input, masked_input, masked_input])

    inception_model_output = inception_partial_model.output
    inception_model_output = GlobalAveragePooling2D()(inception_model_output)
    inception_model_output = Dense(output_shape, activation='softmax')(inception_model_output)
    inception_model = Model(inception_partial_model.input, inception_model_output, name="inception")

    output = inception_model(inception_input)

    model = Model(img_input, outputs=output)

    optimizer = RAdamOptimizer(learning_rate)
    model.compile(optimizer=optimizer, loss='mse', metrics=['accuracy', mse])
    model.name = name

    model.weight_filename = os.path.join(weight_root, f"{name}.h5")
    model.summary_filename = os.path.join(summary_root, f"{name}.txt")
    model.class_weight = class_weight

    if load:
        print('load weights')
        model.load_weights(model.weight_filename[:-3] + ' - copie.h5')

    os.makedirs(os.path.split(model.summary_filename)[0], exist_ok=True)
    with open(model.summary_filename, 'w') as file:
        old = sys.stdout
        sys.stdout = file
        model.summary()
        sys.stdout = old

    return model
