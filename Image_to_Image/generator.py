import numpy as np
import os
import glob
from skimage.transform import resize

from Rignak_DeepLearning.generator import get_add_additional_inputs
from Rignak_DeepLearning.data import read
from Rignak_Misc.path import list_dir, convert_link

BATCH_SIZE = 8
INPUT_SHAPE = (512, 512, 3)
OUTPUT_SHAPE = (32, 32, 1)
INPUT_LABEL = "input"
OUTPUT_LABEL = "output"


def autoencoder_base_generator(root, validation=False, batch_size=BATCH_SIZE, input_shape=INPUT_SHAPE):
    filenames = glob.glob(os.path.join(root, '*/*', '*.png')) + \
                glob.glob(os.path.join(root, '*', '*.png')) + \
                glob.glob(os.path.join(root, '*.png'))
    filenames = np.array(filenames)
    k = 0
    n_files = len(filenames)
    yield None
    while True:
        if validation:
            batch_index = [(k+i)%n_files for i in range(batch_size)]
            k += batch_size
        else:
            batch_index = np.random.randint(0, n_files, size=batch_size)
        batch_path = filenames[batch_index]
        batch_input = np.array([read(path, input_shape) for path in batch_path])
        yield batch_input, batch_input.copy()


def segmenter_base_generator(root, validation=False, batch_size=BATCH_SIZE, input_shape=INPUT_SHAPE, output_shape=OUTPUT_SHAPE,
                             input_label=INPUT_LABEL, output_label=OUTPUT_LABEL, attributes=None):
    input_filenames = np.array(sorted(glob.glob(os.path.join(root, input_label, '*.npy'))+glob.glob(os.path.join(root, input_label, '*.png'))))
    output_filenames = np.array(sorted(glob.glob(os.path.join(root, output_label, '*.npy'))+glob.glob(os.path.join(root, output_label, '*.png'))))
    assert len(input_filenames) == len(output_filenames), f"{len(input_filenames)} - {len(output_filenames)}"

    input_filenames  = input_filenames[:3200] 
    output_filenames  = output_filenames[:3200] 
    [convert_link(filename) for filename in input_filenames[:3200] if filename.endswith('.lnk')]
    [convert_link(filename) for filename in output_filenames if filename.endswith('.lnk')]

    add_additional_inputs = get_add_additional_inputs(root, attributes)

    k = 0
    n_files = len(input_filenames)
    yield None
    while True:
        if validation:
            batch_index = [(k+i)%n_files for i in range(batch_size)]
            k += batch_size
        else:
            batch_index = np.random.randint(0, n_files, size=batch_size)
        batch_input_path = input_filenames[batch_index]
        batch_input = np.array([read(path, input_shape) for path in batch_input_path])

        batch_output_path = output_filenames[batch_index]
        batch_output = np.array([read(path, output_shape) for path in batch_output_path])
        batch_output[batch_output > 0 ] = 1

        batch_input = add_additional_inputs(batch_input, batch_input_path)
        yield batch_input, batch_output


def saliency_base_generator(root, input_shape=INPUT_SHAPE, batch_size=BATCH_SIZE, output_shape=OUTPUT_SHAPE,
                            folders=None, attributes=None):
    folders = list_dir(root) if folders is None else [os.path.join(root, folder) for folder in
                                                      folders[1:-1].split(', ')]
    if len(folders) == 2:
        colors = np.array([[0], [1]])
    elif len(folders) > 2:
        colors = np.eye(len(folders), dtype=np.int) * 1
    else:
        raise ValueError(f'Number of folders (currently {len(folders)}) should be equal or greater than 2')

    mapping = {os.path.join(folder, filename): colors[i]
               for (i, folder) in enumerate(folders)
               for filename in os.listdir(folder)}

    label_to_filename = {folder: [os.path.join(folder, filename)
                                  for filename in os.listdir(folder)]
                         for folder in folders
                         }

    [convert_link(filename) for filename in mapping.keys() if filename.endswith('.lnk')]
    add_additional_inputs = get_add_additional_inputs(root, attributes)

    yield None
    while True:
        batch_labels = np.random.choice(folders, size=batch_size)
        batch_path = [np.random.choice(label_to_filename[label]) for label in batch_labels]

        batch_input = np.array([read(path, input_shape=input_shape) for path in batch_path])
        batch_output = np.zeros((batch_size, output_shape[0], output_shape[1], output_shape[2]))

        for i, input_filename in enumerate(batch_path):
            batch_output[i, :, :] = mapping[input_filename]
            
        batch_input = add_additional_inputs(batch_input, batch_path)
        yield batch_input, batch_output


def thumbnail_base_generator(root, batch_size=BATCH_SIZE, input_shape=INPUT_SHAPE,
                             input_label=INPUT_LABEL, output_label=OUTPUT_LABEL,
                             scaling=1):
    input_filenames = [os.path.join(root, input_label, filename)
                       for filename in os.listdir(os.path.join(root, input_label))]
    output_filenames = [os.path.join(root, output_label, filename)
                        for filename in os.listdir(os.path.join(root, output_label))]
    first_output = read(output_filenames[0])
    yield None
    while True:
        batch_input = np.zeros((batch_size, input_shape[0], input_shape[1], input_shape[2]))
        batch_output = np.zeros((batch_size, input_shape[0], input_shape[1], first_output.shape[-1]))

        filenames_index = np.random.randint(0, len(input_filenames), size=batch_size)
        for i, filename_index in enumerate(filenames_index):
            input_ = read(input_filenames[filename_index])
            output = read(output_filenames[filename_index])
            if scaling != 1 and input_.shape[0] * scaling > input_shape[0] \
                    and input_.shape[1] * scaling > input_shape[1]:
                input_ = resize(input_, (int(input_.shape[0] * scaling), int(input_.shape[1] * scaling)))
                output = resize(output, (int(output.shape[0] * scaling), int(output.shape[1] * scaling)))

            if input_shape[0] == input_.shape[0]:
                x_offset = 0
            else:
                x_offset = np.random.randint(0, input_.shape[0] - input_shape[0])
            if input_shape[1] == input_.shape[1]:
                y_offset = 0
            else:
                y_offset = np.random.randint(0, input_.shape[1] - input_shape[1])
            batch_input[i] = input_[x_offset:x_offset + input_shape[0], y_offset:y_offset + input_shape[1]]
            batch_output[i] = output[x_offset:x_offset + input_shape[0], y_offset:y_offset + input_shape[1]]
        yield batch_input, batch_output
