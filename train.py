import sys
import os

from keras.callbacks import ModelCheckpoint

from Rignak_DeepLearning.data import get_dataset_roots
from Rignak_DeepLearning.normalization import intensity_normalization
from Rignak_DeepLearning.noise import get_uniform_noise_function
from Rignak_DeepLearning.callbacks import HistoryCallback, ExampleCallback, ConfusionCallback
from Rignak_DeepLearning.Autoencoders.flat import import_model as import_flat_model
from Rignak_DeepLearning.Autoencoders.unet import import_model as import_unet_model
from Rignak_DeepLearning.Categorizer.flat import import_model as import_categorizer
from Rignak_DeepLearning.generator import autoencoder_generator, categorizer_generator, saliency_generator
from Rignak_DeepLearning.generator import normalize_generator as normalize, augment_generator as augment

"""
>>> python train.py unet autoencoder fav-rignak name=fav-rignak
>>> python train.py flat saliency open_eyes name=open_eyes
>>> python train.py flat saliency eye_color name=eye_color
>>> python train.py flat categorizer eye_color name=eye_color
OK >>> python train.py flat categorizer mnist name=mnist
"""

INPUT_SHAPE = (256, 256, 3)
BATCH_SIZE = 8
TYPE = 'autoencoder'
NAME = ''

STEPS_PER_EPOCH = 1000
VALIDATION_STEPS = 100
EPOCHS = 1000


def parse_input(argvs, input_shape=INPUT_SHAPE, batch_size=BATCH_SIZE, conv_layers=None, name=NAME):
    type_ = argvs[2]

    dataset = argvs[3]
    if dataset == 'mnist':
        from Rignak_DeepLearning.OCR.mnist_settings import BATCH_SIZE, INPUT_SHAPE, CONV_LAYERS
        input_shape = INPUT_SHAPE
        batch_size = BATCH_SIZE
        conv_layers = CONV_LAYERS

    for argv in argvs[4:]:
        if argv.startswith('input_shape='):
            shape = int(argv.replace('input_shape=', ''))
            input_shape = (shape, shape, 3)
        elif argv.startswith('batch_size='):
            batch_size = int(argv.replace('batch_size=', ''))
        elif argv.startswith('type='):
            type_ = argv.replace('type=', '')
        elif argv.startswith('name='):
            name = argv.replace('name=', '')
        else:
            raise NameError

    normalization_function = intensity_normalization()[0]
    noise_function = get_uniform_noise_function()

    # choose the generators
    train_folder, val_folder = get_dataset_roots(type_, dataset=dataset)
    if type_ == 'saliency':
        train_generator = saliency_generator(train_folder, input_shape=input_shape, batch_size=batch_size)
        val_generator = saliency_generator(val_folder, input_shape=input_shape, batch_size=batch_size)
        if len(os.listdir(train_folder)) == 2:
            output_canals = 1
        else:
            output_canals = len(os.listdir(train_folder))
    elif type_ == 'autoencoder':
        train_generator = autoencoder_generator(train_folder, input_shape=input_shape, batch_size=batch_size)
        val_generator = autoencoder_generator(val_folder, input_shape=input_shape, batch_size=batch_size)
        output_canals = 3
    elif type_ == 'categorizer':
        train_generator = categorizer_generator(train_folder, input_shape=input_shape, batch_size=batch_size)
        val_generator = categorizer_generator(val_folder, input_shape=input_shape, batch_size=batch_size)
        callback_generator = categorizer_generator(val_folder, input_shape=input_shape, batch_size=batch_size)
        labels = os.listdir(train_folder)
        output_canals = len(labels)
    else:
        raise NameError

    # choose the model
    if argvs[1] == 'unet':
        model = import_unet_model(input_shape=input_shape, canals=output_canals, name=name)
    elif argvs[1] == 'flat' and type_ != 'categorizer':
        model = import_flat_model(input_shape=input_shape, canals=output_canals, name=name)
    elif argvs[1] == 'flat' and type_ == 'categorizer':
        if conv_layers:
            model = import_categorizer(output_canals, input_shape=input_shape, name=name)
        else:
            model = import_categorizer(output_canals, input_shape=input_shape, name=name, conv_layers=conv_layers)
    else:
        raise NameError

    # choose the data augmentation, normalization and callbacks
    if type_ in ['saliency', 'autoencoder']:
        train_generator = normalize(augment(train_generator, noise_function=noise_function, apply_on_output=True),
                                    normalization_function, apply_on_output=True)
        val_generator = normalize(augment(val_generator, noise_function=noise_function, apply_on_output=True),
                                  normalization_function, apply_on_output=True)
        callbacks = [ModelCheckpoint(model.weight_filename, save_best_only=True),
                     HistoryCallback(),
                     ExampleCallback(next(val_generator))]
    else:
        train_generator = normalize(augment(train_generator, noise_function=noise_function, apply_on_output=False),
                                    normalization_function, apply_on_output=False)
        val_generator = normalize(augment(val_generator, noise_function=noise_function, apply_on_output=False),
                                  normalization_function, apply_on_output=False)
        callback_generator = normalize(
            augment(callback_generator, noise_function=noise_function, apply_on_output=False),
            normalization_function, apply_on_output=False)
        callbacks = [ModelCheckpoint(model.weight_filename, save_best_only=True),
                     HistoryCallback(),
                     ConfusionCallback(callback_generator, labels)]

    return model, train_generator, val_generator, callbacks


def train(model, train_generator, val_generator, callbacks, steps_per_epoch=STEPS_PER_EPOCH,
          validation_steps=VALIDATION_STEPS, epochs=EPOCHS):
    model.fit_generator(generator=train_generator,
                        validation_data=val_generator,
                        verbose=1,
                        steps_per_epoch=steps_per_epoch,
                        validation_steps=validation_steps,
                        epochs=epochs,
                        callbacks=callbacks)


if __name__ == '__main__':
    model, train_generator, val_generator, callbacks = parse_input(sys.argv)
    train(model, train_generator, val_generator, callbacks)
