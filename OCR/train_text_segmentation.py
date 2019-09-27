import os
import numpy as np
from PIL import Image

from keras.callbacks import ModelCheckpoint

from Rignak_Misc.path import get_local_file
from Rignak_DeepLearning.noise import get_uniform_noise_function
from Rignak_DeepLearning.normalization import intensity_normalization
from Rignak_DeepLearning.generator import thumbnail_generator

from Rignak_DeepLearning.Autoencoders.unet import import_model
from Rignak_DeepLearning.callbacks import HistoryCallback, ExampleCallback
from Rignak_DeepLearning.generator import normalize_generator as normalize, augment_generator as augment

SEGMENTATION_DATASET_ROOT = get_local_file(__file__, os.path.join('datasets', 'text_segmentation'))
MODEL_FILENAME = get_local_file(__file__, 'text_segmentation.h5')

BATCH_SIZE = 16
INPUT_SHAPE = (128, 128, 3)
NEURON_BASIS = 4
STEPS_PER_EPOCH = 500
VALIDATION_STEPS = 25
EPOCHS = 500


def create_dataset(dataset_root=SEGMENTATION_DATASET_ROOT):
    training_inputs_root = os.path.join(dataset_root, 'training', 'x')
    training_outputs_root = os.path.join(dataset_root, 'training', 'y')
    validation_inputs_root = os.path.join(dataset_root, 'validation', 'x')
    validation_outputs_root = os.path.join(dataset_root, 'validation', 'y')

    training_dataset = []
    validation_dataset = []

    for input_filename, output_filename in zip(os.listdir(training_inputs_root), os.listdir(training_outputs_root)):
        with Image.open(os.path.join(training_outputs_root, output_filename)) as output_image:
            with Image.open(os.path.join(training_inputs_root, input_filename)) as input_image:
                training_dataset.append((np.array(input_image), np.array(output_image)[:, :, :1]))

    for input_filename, output_filename in zip(os.listdir(validation_inputs_root), os.listdir(validation_outputs_root)):
        with Image.open(os.path.join(validation_outputs_root, output_filename)) as output_image:
            with Image.open(os.path.join(validation_inputs_root, input_filename)) as input_image:
                validation_dataset.append((np.array(input_image), np.array(output_image)[:, :, :1]))

    return training_dataset, validation_dataset


def main(batch_size=BATCH_SIZE, input_shape=INPUT_SHAPE, neuron_basis=NEURON_BASIS, steps_per_epoch=STEPS_PER_EPOCH,
         validation_steps=VALIDATION_STEPS, epochs=EPOCHS, model_filename=MODEL_FILENAME):
    training_dataset, validation_dataset = create_dataset()
    noise_function = get_uniform_noise_function()
    normalization_function = intensity_normalization()[0]

    training_generator = thumbnail_generator(training_dataset, batch_size=batch_size, shape=input_shape)
    training_generator = normalize(augment(training_generator, noise_function=noise_function, apply_on_output=True),
                                   normalization_function, apply_on_output=True)

    validation_generator = thumbnail_generator(validation_dataset, batch_size, shape=input_shape)
    validation_generator = normalize(augment(validation_generator, noise_function=noise_function, apply_on_output=True),
                                     normalization_function, apply_on_output=True)

    callbacks = [ModelCheckpoint(model_filename, save_best_only=True, save_weights_only=True),
                 HistoryCallback(), ExampleCallback(next(validation_generator))]
    model = import_model(input_shape=input_shape, neuron_basis=neuron_basis)
    model.fit_generator(generator=training_generator, validation_data=validation_generator,
                        verbose=2, steps_per_epoch=steps_per_epoch, validation_steps=validation_steps,
                        epochs=epochs, callbacks=callbacks)


if __name__ == '__main__':
    main()
