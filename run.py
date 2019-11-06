import os
import numpy as np
from tqdm import tqdm

from keras.models import load_model as keras_load_model

from Rignak_Misc.path import get_local_file
from Rignak_DeepLearning.data import read

MODELS_FOLDER = get_local_file(__file__, os.path.join('_outputs', 'models'))
BATCH_SIZE = 16


def load_model(model_filename, models_folder=MODELS_FOLDER):
    return keras_load_model(os.path.join(models_folder, model_filename))


def run_batch(array, model, batch_size=BATCH_SIZE, normalization = None):
    output_shape = model.layers[-1].output_shape[-1]
    output = np.zeros((array.shape[0], output_shape))
    for batch_i in tqdm(range(array.shape[0] // batch_size + 1)):
        batch = array[batch_i * batch_size:(batch_i + 1) * batch_size]
        if normalization is not None:
            batch = normalization(batch)
        output[batch_i * batch_size:(batch_i + 1) * batch_size] = model.predict(batch)
    return output


def run_batch_on_filenames(filenames, model, batch_size=BATCH_SIZE, normalization = None):
    output_shape = model.layers[-1].output_shape[-1]
    input_shape = model.layers[0].input_shape[-3:]
    output = np.zeros((len(filenames), output_shape))
    for batch_i in tqdm(range(len(filenames) // batch_size+1)):
        batch_filename = filenames[batch_i * batch_size:(batch_i + 1) * batch_size]
        if not batch_filename:
            break
        batch = np.array([read(filename, input_shape) for filename in batch_filename])
        if normalization is not None:
            batch = normalization(batch)
        output[batch_i * batch_size:(batch_i + 1) * batch_size] = model.predict(batch)
    return output
