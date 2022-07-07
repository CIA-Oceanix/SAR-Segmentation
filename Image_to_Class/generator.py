import numpy as np
import os

from Rignak_DeepLearning.generator import get_add_additional_inputs
from Rignak_DeepLearning.data import read
from Rignak_Misc.path import list_dir, convert_link

BATCH_SIZE = 8
INPUT_SHAPE = (256, 256, 3)


def categorizer_base_generator(root, batch_size=BATCH_SIZE, validation=False, input_shape=INPUT_SHAPE, folders=None, attributes=None):
    if not os.path.exists(root) and os.path.exists(root + '.lnk'):
        root = convert_link(root +'.lnk')
    folders = folders[1:-1].split(', ') if isinstance(folders, str) else folders
    folders = list_dir(root) if folders is None else [os.path.join(root, folder) for folder in folders]
    folders = [convert_link(folder) if folder.endswith('.lnk') else folder for folder in folders]

    filename_to_hot_label = {
        os.path.join(folder, filename): make_categorizer_output(folders.index(folder), len(folders))
        for folder in folders
        for filename in os.listdir(folder)
    }

    label_to_filename = {
        folder: [os.path.join(folder, filename)
                 for filename in os.listdir(folder) if '.png' in filename]
        for folder in folders
    }

    [convert_link(filename) for filename in filename_to_hot_label.keys() if filename.endswith('.lnk')]
    add_additional_inputs = get_add_additional_inputs(root, attributes)

    yield None
    while True:
        batch_labels = np.random.choice(folders, size=batch_size)
        batch_input_path = np.array([np.random.choice(label_to_filename[label]) for label in batch_labels])
        batch_input = np.array([read(path, input_shape=input_shape) for path in batch_input_path])
        batch_output = np.array([filename_to_hot_label[filename] for filename in batch_input_path])
        batch_input = add_additional_inputs(batch_input, batch_input_path)
        yield batch_input, batch_output


def make_categorizer_output(index, label_number):
    output = np.zeros(label_number)
    output[index] = 1
    return output
