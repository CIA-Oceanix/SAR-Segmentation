import os
import fire

import Rignak_DeepLearning.deprecation_warnings

from keras.callbacks import ModelCheckpoint

from Rignak_DeepLearning.data import get_dataset_roots
from Rignak_DeepLearning.normalization import intensity_normalization
from Rignak_DeepLearning.noise import get_uniform_noise_function
from Rignak_DeepLearning.callbacks import HistoryCallback, AutoencoderExampleCallback, ConfusionCallback, \
    ClassificationExampleCallback
from Rignak_DeepLearning.Autoencoders.flat import import_model as import_flat_model
from Rignak_DeepLearning.Autoencoders.unet import import_model as import_unet_model
from Rignak_DeepLearning.Categorizer.flat import import_model as import_categorizer
from Rignak_DeepLearning.Categorizer.inception import import_model_v3 as InceptionV3
from Rignak_DeepLearning.BiOutput.flat import import_model as import_bimode
from Rignak_DeepLearning.BiOutput.generator import generator as bimode_generator, \
    normalize_generator as bimode_normalize, augment_generator as bimode_augment
from Rignak_DeepLearning.BiOutput.callbacks import ExampleCallback as BimodeExampleCallback
from Rignak_DeepLearning.BiOutput.callbacks import HistoryCallback as BimodeHistoryCallback
from Rignak_DeepLearning.generator import autoencoder_generator, categorizer_generator, saliency_generator, \
    thumbnail_generator as thumb_generator, normalize_generator, augment_generator
from Rignak_DeepLearning.config import get_config

"""
>>> python train.py autoencoder fav-rignak 
>>> python train.py saliency open_eyes
>>> python train.py saliency eye_color
>>> python train.py categorizer waifu
>>> python train.py mnist mnist batch_size=256
>>> python train.py style_transfer colorization 
>>> python train.py bimode waifu 
"""

BATCH_SIZE = 8
TRAINING_STEPS = 2500
VALIDATION_STEPS = 250
EPOCHS = 2000

DEFAULT_INPUT_SHAPE = (256, 256, 3)
DEFAULT_SCALING = 1


def get_generators(config, task, dataset, batch_size, default_input_shape=DEFAULT_INPUT_SHAPE,
                   default_scaling=DEFAULT_SCALING):
    def get_saliency_generators():
        train_generator = saliency_generator(train_folder, input_shape=input_shape, batch_size=batch_size)
        val_generator = saliency_generator(val_folder, input_shape=input_shape, batch_size=batch_size)
        callback_generator = saliency_generator(val_folder, input_shape=input_shape, batch_size=batch_size)
        return train_generator, val_generator, callback_generator, train_folder

    def get_autoencoder_generators():
        train_generator = autoencoder_generator(train_folder, input_shape=input_shape, batch_size=batch_size)
        val_generator = autoencoder_generator(val_folder, input_shape=input_shape, batch_size=batch_size)
        callback_generator = autoencoder_generator(val_folder, input_shape=input_shape, batch_size=batch_size)
        return train_generator, val_generator, callback_generator, train_folder

    def get_categorizer_generators():
        train_generator = categorizer_generator(train_folder, input_shape=input_shape, batch_size=batch_size)
        val_generator = categorizer_generator(val_folder, input_shape=input_shape, batch_size=batch_size)
        callback_generator = categorizer_generator(val_folder, input_shape=input_shape, batch_size=batch_size)
        return train_generator, val_generator, callback_generator, train_folder

    def get_style_transfer_generators():
        train_generator = thumb_generator(train_folder, input_shape=input_shape, batch_size=batch_size, scaling=scaling)
        val_generator = thumb_generator(val_folder, input_shape=input_shape, batch_size=batch_size, scaling=scaling)
        callback_gene = thumb_generator(val_folder, input_shape=input_shape, batch_size=batch_size, scaling=scaling)
        return train_generator, val_generator, callback_gene, train_folder

    def get_bimode_generator():
        train_generator = bimode_generator(train_folder, input_shape=input_shape, batch_size=batch_size)
        val_generator = bimode_generator(val_folder, input_shape=input_shape, batch_size=batch_size)
        callback_generator = bimode_generator(val_folder, input_shape=input_shape, batch_size=batch_size)
        return train_generator, val_generator, callback_generator, train_folder

    input_shape = config[task].get('INPUT_SHAPE', default_input_shape)
    scaling = config[task].get('SCALING', default_scaling)
    train_folder, val_folder = get_dataset_roots(task, dataset=dataset)

    functions = {"saliency": get_saliency_generators,
                 "autoencoder": get_autoencoder_generators,
                 "flat_autoencoder": get_autoencoder_generators,
                 "categorizer": get_categorizer_generators,
                 "inceptionV3": get_categorizer_generators,
                 "style_transfer": get_style_transfer_generators,
                 "bimode": get_bimode_generator,
                 }
    return functions[task]()


def get_data_augmentation(task, train_generator, val_generator, callback_generator):
    def get_im2im_data_augmentation():
        new_train_generator = normalize_generator(augment_generator(train_generator, noise_function=noise_function, apply_on_output=True), normalization_function, apply_on_output=True)
        new_val_generator = normalize_generator(augment_generator(val_generator, noise_function=noise_function, apply_on_output=True), normalization_function, apply_on_output=True)
        new_callback_generator = normalize_generator(augment_generator(callback_generator, noise_function=noise_function, apply_on_output=True), normalization_function, apply_on_output=True)
        return new_train_generator, new_val_generator, new_callback_generator

    def get_bimode_augmentation():
        new_train_generator = bimode_normalize(bimode_augment(train_generator, noise_function=noise_function, apply_on_output=True), normalization_function, apply_on_output=True)
        new_val_generator = bimode_normalize(bimode_augment(val_generator, noise_function=noise_function, apply_on_output=True), normalization_function, apply_on_output=True)
        new_callback_generator = bimode_normalize(bimode_augment(callback_generator, noise_function=noise_function, apply_on_output=True), normalization_function, apply_on_output=True)
        return new_train_generator, new_val_generator, new_callback_generator

    def get_categorizer_augmentation():
        new_train_generator = normalize_generator(augment_generator(train_generator, noise_function=noise_function, apply_on_output=False), normalization_function, apply_on_output=False)
        new_val_generator = normalize_generator(augment_generator(val_generator, noise_function=noise_function, apply_on_output=False), normalization_function, apply_on_output=False)
        new_callback_generator = normalize_generator(augment_generator(callback_generator, noise_function=noise_function, apply_on_output=False), normalization_function, apply_on_output=False)
        return new_train_generator, new_val_generator, new_callback_generator

    normalization_function = intensity_normalization()[0]
    noise_function = get_uniform_noise_function()

    functions = {"style_transfer": get_im2im_data_augmentation,
                 "saliency": get_im2im_data_augmentation,
                 "autoencoder": get_im2im_data_augmentation,
                 "flat_autoencoder": get_im2im_data_augmentation,
                 "categorizer": get_categorizer_augmentation,
                 "inceptionV3": get_categorizer_augmentation,
                 "bimode": get_bimode_augmentation,
                 }
    return functions[task]()


def get_models(config, task, name, train_folder, default_input_shape=DEFAULT_INPUT_SHAPE):
    def get_saliency_model():
        if len(labels) == 2:
            config[task]['OUTPUT_CANALS'] = 1
        else:
            config[task]['OUTPUT_CANALS'] = len(labels)
        model = import_flat_model(name=name, config=config[task])
        model.callback_titles = ['Input', 'Prediction', 'Truth'] + labels
        return model

    def get_autoencoder_model():
        if task == 'flat_autoencoder':
            model = import_flat_model(name=name, config=config[task])
        else:
            model = import_unet_model(name=name, config=config[task])
        model.callback_titles = ['Input', 'Prediction', 'Truth'] + labels
        return model

    def get_categorizer_model():
        if task == 'inceptionV3':
            model = InceptionV3(input_shape, len(labels), name, imagenet=config[task]['IMAGENET'])
        else:
            model = import_categorizer(len(labels), config=config[task], name=name)
        model.labels = labels
        return model

    def get_bimode_model():
        model = import_bimode(config[task]['OUTPUT_CANALS'], labels, config=config[task], name=name)
        model.labels = labels
        return model

    labels = os.listdir(train_folder)
    input_shape = config[task].get('INPUT_SHAPE', default_input_shape)

    functions = {"saliency": get_saliency_model,
                 "autoencoder": get_autoencoder_model,
                 "flat_autoencoder": get_autoencoder_model,
                 "style_transfer": get_autoencoder_model,
                 "categorizer": get_categorizer_model,
                 "inceptionV3": get_categorizer_model,
                 "bimode": get_bimode_model,
                 }
    return functions[task]()


def get_callbacks(task, model, callback_generator):
    def get_im2im_callbacks():
        callbacks = [ModelCheckpoint(model.weight_filename, save_best_only=True),
                     HistoryCallback(),
                     AutoencoderExampleCallback(callback_generator)]
        return callbacks

    def get_bimode_callbacks():
        callbacks = [ModelCheckpoint(model.weight_filename, save_best_only=True),
                     BimodeHistoryCallback(),
                     BimodeExampleCallback(callback_generator),
                     ConfusionCallback(callback_generator, model.labels)]
        return callbacks

    def get_categorizer_callbacks():
        callbacks = [ModelCheckpoint(model.weight_filename, save_best_only=True),
                     HistoryCallback(),
                     ConfusionCallback(callback_generator, model.labels),
                     ClassificationExampleCallback(callback_generator)]
        return callbacks

    functions = {"saliency": get_im2im_callbacks,
                 "autoencoder": get_im2im_callbacks,
                 "flat_autoencoder": get_im2im_callbacks,
                 "style_transfer": get_im2im_callbacks,
                 "categorizer": get_categorizer_callbacks,
                 "inceptionV3": get_categorizer_callbacks,
                 "bimode": get_bimode_callbacks,
                 }
    return functions[task]()


def main(task, dataset, batch_size=BATCH_SIZE, epochs=EPOCHS,
         training_steps=TRAINING_STEPS, validation_steps=VALIDATION_STEPS):
    config = get_config()
    task = config[task]['TASK']

    train_folder, val_folder = get_dataset_roots(task, dataset=dataset)
    train_generator, val_generator, callback_generator, train_dir = get_generators(config, task, dataset, batch_size)
    train_generator, val_generator, callback_generator = get_data_augmentation(task, train_generator, val_generator,
                                                                               callback_generator)
    name = f'{dataset}_{task}'
    model = get_models(config, task, name, train_folder)
    callbacks = get_callbacks(task, model, callback_generator)

    train(model, train_generator, val_generator, callbacks,
          epochs=epochs, training_steps=training_steps, validation_steps=validation_steps)


def train(model, train_generator, val_generator, callbacks, training_steps=TRAINING_STEPS,
          validation_steps=VALIDATION_STEPS, epochs=EPOCHS):
    model.fit_generator(generator=train_generator,
                        validation_data=val_generator,
                        verbose=1,
                        steps_per_epoch=training_steps,
                        validation_steps=validation_steps,
                        epochs=epochs,
                        callbacks=callbacks)


if __name__ == '__main__':
    fire.Fire(main)
