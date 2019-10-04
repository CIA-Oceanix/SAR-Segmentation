import sys
import os
import fire

from keras.callbacks import ModelCheckpoint

from Rignak_DeepLearning.data import get_dataset_roots
from Rignak_DeepLearning.normalization import intensity_normalization
from Rignak_DeepLearning.noise import get_uniform_noise_function
from Rignak_DeepLearning.callbacks import HistoryCallback, AutoencoderExampleCallback, ConfusionCallback, \
    ClassificationExampleCallback
from Rignak_DeepLearning.Autoencoders.flat import import_model as import_flat_model
from Rignak_DeepLearning.Autoencoders.unet import import_model as import_unet_model
from Rignak_DeepLearning.Categorizer.flat import import_model as import_categorizer
from Rignak_DeepLearning.generator import autoencoder_generator, categorizer_generator, saliency_generator, \
    thumbnail_generator, normalize_generator as normalize, augment_generator as augment
from Rignak_DeepLearning.config import get_config

"""
>>> python train.py autoencoder fav-rignak 
>>> python train.py saliency open_eyes
>>> python train.py saliency eye_color
>>> python train.py categorizer waifu
>>> python train.py mnist mnist batch_size=256
>>> python train.py style_transfer colorization 
"""

BATCH_SIZE = 8

STEPS_PER_EPOCH = 10000
VALIDATION_STEPS = 1000
EPOCHS = 1000


def main(task, dataset, batch_size=BATCH_SIZE):
    """
    Train a network

    :param task: the type of neural network,
    either "autoencoder", "saliency", "mnist", "categorizer", "style_transfer" or "speech_categorization"
    :param dataset: name of the folder containing the images
    :param batch_size: size of each batch
    :return:
    """
    config = get_config(task)
    task = config['TASK']
    normalization_function = intensity_normalization()[0]
    noise_function = get_uniform_noise_function()
    name = dataset

    train_folder, val_folder = get_dataset_roots(task, dataset=dataset)
    if task == 'saliency':
        train_generator = saliency_generator(train_folder, input_shape=config['INPUT_SHAPE'], batch_size=batch_size)
        val_generator = saliency_generator(val_folder, input_shape=config['INPUT_SHAPE'], batch_size=batch_size)
        callback_generator = saliency_generator(val_folder, input_shape=config['INPUT_SHAPE'], batch_size=batch_size)

        if len(os.listdir(train_folder)) == 2:
            config['OUTPUT_CANALS'] = 1
        else:
            config['OUTPUT_CANALS'] = len(os.listdir(train_folder))
        model = import_flat_model(name=name, config=config)

    elif task == 'autoencoder':
        train_generator = autoencoder_generator(train_folder, input_shape=config['INPUT_SHAPE'], batch_size=batch_size)
        val_generator = autoencoder_generator(val_folder, input_shape=config['INPUT_SHAPE'], batch_size=batch_size)
        callback_generator = autoencoder_generator(val_folder, input_shape=config['INPUT_SHAPE'], batch_size=batch_size)

        model = import_unet_model(name=name, config=config)

    elif task == 'categorizer':
        train_generator = categorizer_generator(train_folder, input_shape=config['INPUT_SHAPE'], batch_size=batch_size)
        val_generator = categorizer_generator(val_folder, input_shape=config['INPUT_SHAPE'], batch_size=batch_size)

        callback_generator = categorizer_generator(val_folder, input_shape=config['INPUT_SHAPE'], batch_size=batch_size)
        labels = os.listdir(train_folder)
        output_canals = len(labels)
        print('here', output_canals)
        model = import_categorizer(output_canals, config=config, name=name)
        model.labels = labels

    elif task == 'style_transfer':
        train_generator = thumbnail_generator(train_folder, input_shape=config['INPUT_SHAPE'], batch_size=batch_size,
                                              scaling=config['SCALING'])
        val_generator = thumbnail_generator(val_folder, input_shape=config['INPUT_SHAPE'], batch_size=batch_size,
                                            scaling=config['SCALING'])
        callback_generator = thumbnail_generator(val_folder, input_shape=config['INPUT_SHAPE'], batch_size=batch_size,
                                                 scaling=config['SCALING'])

        model = import_unet_model(name=name, config=config)

    else:
        raise NameError

    # choose the data augmentation, normalization and callbacks
    if task in ['saliency', 'autoencoder', 'style_transfer']:
        train_generator = normalize(augment(train_generator, noise_function=noise_function, apply_on_output=True),
                                    normalization_function, apply_on_output=True)
        val_generator = normalize(augment(val_generator, noise_function=noise_function, apply_on_output=True),
                                  normalization_function, apply_on_output=True)
        callback_generator = normalize(augment(callback_generator, noise_function=noise_function, apply_on_output=True),
                                       normalization_function, apply_on_output=True)
        callbacks = [ModelCheckpoint(model.weight_filename, save_best_only=True),
                     HistoryCallback(),
                     AutoencoderExampleCallback(callback_generator)]
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
                     ConfusionCallback(callback_generator, labels),
                     ClassificationExampleCallback(callback_generator)]

    train(model, train_generator, val_generator, callbacks)


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
    fire.Fire(main)
