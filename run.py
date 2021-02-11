import os
import numpy as np
import tqdm.auto as tqdm

from tensorflow.keras.models import load_model as keras_load_model

from Rignak_Misc.path import get_local_file
from Rignak_DeepLearning.data import read

MODELS_FOLDER = get_local_file(__file__, os.path.join('_outputs', 'models'))
BATCH_SIZE = 16


def load_model(model_filename, models_folder=MODELS_FOLDER):
    return keras_load_model(os.path.join(models_folder, model_filename))


def run_batch(array, model, batch_size=BATCH_SIZE, normalizer=None, use_tqdm=False):
    output_shape = model.layers[-1].output_shape[-1]
    output = np.zeros((array.shape[0], output_shape))

    batch_range = range(array.shape[0] // batch_size + 1)
    if use_tqdm:
        batch_range = tqdm.tqdm(batch_range)
    for batch_i in batch_range:
        batch = array[batch_i * batch_size:(batch_i + 1) * batch_size]
        if normalizer is not None:
            batch = normalizer(batch)
        if batch_i * batch_size != output.shape[0]:
            output[batch_i * batch_size:(batch_i + 1) * batch_size] = model.predict(batch)
    return output


def run_batch_on_filenames(filenames, model, batch_size=BATCH_SIZE, normalizer=None, use_tqdm=True):
    output_shape = model.output.shape[1:]
    input_shape = model.layers[0].input_shape[-3:]
    output = np.zeros([len(filenames)] + list(output_shape))

    batch_range = range(len(filenames) // batch_size + 1)
    if use_tqdm:
        batch_range = tqdm.tqdm(batch_range)
    for batch_i in batch_range:
        batch_filename = filenames[batch_i * batch_size:(batch_i + 1) * batch_size]
        if not batch_filename:
            break
        batch = np.array([read(filename, input_shape) for filename in batch_filename])
        if normalizer is not None:
            batch = normalizer(batch)
        output[batch_i * batch_size:(batch_i + 1) * batch_size] = model.predict(batch)
    return output


def run_batch_on_filenames_with_multiple_inputs(filenames, model, normalized_inputs, batch_size=BATCH_SIZE):
    output_shape = model.output.shape[1:]
    input_shape = model.layers[0].input_shape[-3:]
    output = np.zeros([len(filenames)] + list(output_shape))

    batch_range = range(len(filenames) // batch_size + 1)
    batch_range = tqdm.tqdm(batch_range)
    for batch_i in batch_range:
        batch_filename = filenames[batch_i * batch_size:(batch_i + 1) * batch_size]
        if not batch_filename:
            break
        batch = np.array([read(filename, input_shape) for filename in batch_filename])
        batch = [batch, np.array([normalized_inputs[os.path.split(filename)[-1]] for filename in batch_filename])]
        output[batch_i * batch_size:(batch_i + 1) * batch_size] = model.predict(batch)
    return output
