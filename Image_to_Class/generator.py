import numpy as np
import os
import glob

from Rignak_DeepLearning.data import read
from Rignak_Misc.path import list_dir, convert_link

BATCH_SIZE = 8
INPUT_SHAPE = (256, 256, 3)
ZOOM = 0.0
ROTATION = 0


def categorizer_base_generator(root, batch_size=BATCH_SIZE, input_shape=INPUT_SHAPE, folders=None):
    folders = list_dir(root) if folders is None else [os.path.join(root, folder) for folder in
                                                      folders[1:-1].split(', ')]
    filename_to_hot_label = {
        os.path.join(folder, filename): make_categorizer_output(folders.index(folder), len(folders))
        for folder in folders
        for filename in os.listdir(folder)
    }

    label_to_filename = {folder: [os.path.join(folder, filename)
                                  for filename in os.listdir(folder)]
                         for folder in folders
                         }

    [convert_link(filename) for filename in filename_to_hot_label.keys() if filename.endswith('.lnk')]

    yield None
    while True:
        batch_labels = np.random.choice(folders, size=batch_size)
        batch_path = np.array([np.random.choice(label_to_filename[label]) for label in batch_labels])
        batch_input = np.array([read(path, input_shape=input_shape) for path in batch_path])
        batch_output = np.array([filename_to_hot_label[filename] for filename in batch_path])
        yield batch_input, batch_output


def regressor_base_generator(root, batch_size=BATCH_SIZE, input_shape=INPUT_SHAPE):
    input_filenames = np.array(glob.glob(os.path.join(root, '*', '*.*')) + glob.glob(os.path.join(root, '*.*')))
    output_filenames = np.array([os.path.splitext(filename)[0] + ".npy" for filename in input_filenames])

    yield None
    while True:
        filenames_index = np.random.randint(0, len(input_filenames), size=batch_size)

        batch_input = np.array([read(filename, input_shape) for filename in input_filenames[filenames_index]])
        batch_output = np.array([np.load(filename)[0] for filename in output_filenames[filenames_index]])
        yield batch_input, batch_output


def make_categorizer_output(index, label_number):
    output = np.zeros(label_number)
    output[index] = 1
    return output
