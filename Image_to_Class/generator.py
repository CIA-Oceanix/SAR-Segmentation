import numpy as np
import os
import glob
import json

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


def regressor_base_generator(root, attributes, batch_size=BATCH_SIZE, input_shape=INPUT_SHAPE):
    if isinstance(attributes, str):
        while ' ' in attributes:
            attributes = attributes.replace(' ', '')
        if attributes.startswith('(') and attributes.endswith(')'):
            attributes = attributes[1:-1]
        attributes = attributes.split(',')

    filenames = np.array(sorted(glob.glob(os.path.join(root, '*.*'))))
    with open(os.path.join(os.path.split(root)[0], 'output.json')) as json_file:
        data = json.load(json_file)

    checked_filenames = []
    for filename in filenames:
        if filename.endswith('.lnk'):
            filename = convert_link(filename)
        # if os.path.split(filename)[-1] not in data:
        #     print(filename, os.path.split(filename)[-1])
        if all([os.path.split(filename)[-1] in data
                and attribute in data[os.path.split(filename)[-1]]
                and not np.isnan(data[os.path.split(filename)[-1]][attribute])
                for attribute in attributes]):
            checked_filenames.append(filename)
    filenames = np.array(checked_filenames)

    means = np.mean([[data[os.path.split(filename)[-1]][attribute]
                      for attribute in attributes]
                     for filename in filenames], axis=0)
    stds = np.std([[data[os.path.split(filename)[-1]][attribute]
                    for attribute in attributes]
                   for filename in filenames], axis=0) * 2

    print(f'The attributes were defined for {len(filenames)} files')
    print('MEANS:', means)
    print('STDS:', stds)
    yield tuple(means), tuple(stds)
    while True:
        filenames_index = np.random.randint(0, len(filenames), size=batch_size)

        batch_filenames = filenames[filenames_index]
        batch_input = np.array([read(filename, input_shape) for filename in batch_filenames])
        batch_output = np.array([[data[os.path.split(filename)[-1]][attribute]
                                  for attribute in attributes]
                                 for filename in batch_filenames])
        batch_output = (batch_output - means) / stds
        yield batch_input, batch_output


def make_categorizer_output(index, label_number):
    output = np.zeros(label_number)
    output[index] = 1
    return output
