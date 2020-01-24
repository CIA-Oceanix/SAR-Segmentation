import numpy as np
import cv2
import os
from skimage.transform import resize
import glob

from Rignak_DeepLearning.data import read

BATCH_SIZE = 8
INPUT_SHAPE = (256, 256, 3)
ZOOM = 0.0
ROTATION = 0


def autoencoder_generator(root, batch_size=BATCH_SIZE, input_shape=INPUT_SHAPE):
    filenames = glob.glob(os.path.join(root, '*', '*.png')) + glob.glob(os.path.join(root, '*.png'))
    while True:
        batch_path = np.random.choice(filenames, size=batch_size)
        batch_input = np.array([read(path, input_shape) for path in batch_path])
        yield batch_input, batch_input.copy()


def make_categorizer_output(index, label_number):
    output = np.zeros(label_number)
    output[index] = 1
    return output


def categorizer_generator(root, batch_size=BATCH_SIZE, input_shape=INPUT_SHAPE):
    folders = [folder for folder in os.listdir(root) if os.path.isdir(os.path.join(root, folder))]
    tags = {os.path.join(root, folder, filename): make_categorizer_output(folders.index(folder), len(folders))
            for folder in folders
            for filename in os.listdir(os.path.join(root, folder))}
    filenames = list(tags.keys())

    while True:
        batch_path = np.random.choice(filenames, size=batch_size)

        batch_input = np.array([read(path, input_shape=input_shape) for path in batch_path])
        batch_output = np.array([tags[filename] for filename in batch_path])
        yield batch_input, batch_output


def saliency_generator(root, input_shape=INPUT_SHAPE, batch_size=BATCH_SIZE):
    folders = os.listdir(root)
    if len(folders) == 2:
        colors = np.array([[0], [255]])
    elif len(folders) > 2:
        colors = np.eye(len(folders), dtype=np.int) * 255
    else:
        raise ValueError(f'Number of folders (currently {len(folders)})should be more than 2')

    output_shape = (input_shape[0], input_shape[1], colors.shape[-1])
    mapping = {os.path.join(root, folder, filename): colors[i]
               for (i, folder) in enumerate(folders)
               for filename in os.listdir(os.path.join(root, folder))}
    filenames = list(mapping.keys())

    input_canals = read(filenames[0]).shape[-1]
    while True:
        selected_filenames = np.random.choice(filenames, size=batch_size)

        batch_input = np.zeros((batch_size, input_shape[0], input_shape[1], input_canals))
        batch_output = np.zeros((batch_size, output_shape[0], output_shape[1], output_shape[2]))

        for i, selected_filename in enumerate(selected_filenames):
            batch_input[i] = read(selected_filename, input_shape=input_shape)
            batch_output[i, :, :] = mapping[selected_filename]

        yield batch_input, batch_output


def normalize_generator(generator, normalizer, apply_on_output=False):
    while True:
        batch_input, batch_output = next(generator)
        batch_input = normalizer(batch_input)
        if apply_on_output:
            batch_output = normalizer(batch_output)
        yield batch_input, batch_output


def augment_generator(generator, zoom_factor=ZOOM, rotation=ROTATION, noise_function=None, apply_on_output=False):
    while True:
        batch_input, batch_output = next(generator)
        input_shape = batch_input.shape[1:3]

        angles = (np.random.random(size=batch_input.shape[0]) - 0.5) * rotation
        zooms = 1 + (np.random.random(size=batch_input.shape[0]) - 0.5) * zoom_factor * 2
        h_flips = np.random.randint(0, 2, size=batch_input.shape[0])

        for i, (input_, output, angle, zoom, h_flip) in \
                enumerate(zip(batch_input, batch_output, angles, zooms, h_flips)):
            if h_flip:
                input_ = input_[:, ::-1]
                if apply_on_output:
                    output = output[:, ::-1]

            rotation_matrix = cv2.getRotationMatrix2D((input_shape[0] // 2, input_shape[1] // 2), angle, zoom)
            if batch_input.shape[-1] != 1:
                batch_input[i] = cv2.warpAffine(input_, rotation_matrix, input_shape[:2])
            else:
                batch_input[i, :, :, 0] = cv2.warpAffine(input_, rotation_matrix, input_shape[:2])

            if apply_on_output:
                if batch_output.shape[1] == 2:
                    if batch_output.shape[-1] == 1:
                        batch_output[i, 1, :, :, 0] = cv2.warpAffine(output, rotation_matrix, input_shape[:2])
                    else:
                        batch_output[i, 1] = cv2.warpAffine(output, rotation_matrix, input_shape[:2])
                else:
                    if batch_output.shape[-1] == 1:
                        batch_output[i, :, :, 0] = cv2.warpAffine(output, rotation_matrix, input_shape[:2])
                    else:
                        batch_output[i] = cv2.warpAffine(output, rotation_matrix, input_shape[:2])

        if noise_function is not None:
            batch_input = noise_function(batch_input)
        yield batch_input, batch_output


def fake_generator(dataset, batch_size=BATCH_SIZE):
    inputs = np.stack(dataset[:, 0])
    outputs = np.stack(dataset[:, 1])
    i = 0
    while True:
        batch_input = np.zeros((batch_size, inputs.shape[1], inputs.shape[2], inputs.shape[3]))
        batch_output = np.zeros((batch_size, outputs.shape[1]))
        for j in range(batch_size):
            batch_input[j] = inputs[i % inputs.shape[0]]
            batch_output[j] = outputs[i % outputs.shape[0]]
            i += 1
        yield batch_input, batch_output


def thumbnail_generator(root, batch_size=BATCH_SIZE, input_shape=INPUT_SHAPE, input_root='input', output_root='output',
                        scaling=1):
    input_filenames = [os.path.join(root, input_root, filename)
                       for filename in os.listdir(os.path.join(root, input_root))]
    output_filenames = [os.path.join(root, output_root, filename)
                        for filename in os.listdir(os.path.join(root, output_root))]
    first_output = read(output_filenames[0])
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


def rotsym_augmentor(generator):
    while True:
        batch_input, batch_output = next(generator)
        symmetries = np.random.randint(0, 2, size=(batch_input.shape[0], 2)) * 2 - 1
        rotations = np.random.randint(0, 4, size=(batch_input.shape[0]))
        for i, ((vertical_symmetry, horizontal_symmetry), rotation) in enumerate(zip(symmetries, rotations)):
            batch_input[i] = np.rot90(batch_input[i, ::vertical_symmetry, ::vertical_symmetry], k=rotation)
            # batch_output[i] = np.rot90(batch_output[i, ::vertical_symmetry, ::vertical_symmetry], k=rotation)
        yield batch_input, batch_output


def regressor_generator(root, batch_size=BATCH_SIZE, input_shape=INPUT_SHAPE):
    input_filenames = np.array(glob.glob(os.path.join(root, '*', '*.png')) + glob.glob(os.path.join(root, '*.png')))
    output_filenames = np.array([os.path.splitext(filename)[0] + ".npy" for filename in input_filenames])
    while True:
        filenames_index = np.random.randint(0, len(input_filenames), size=batch_size)

        batch_input = np.array([read(filename, input_shape) for filename in input_filenames[filenames_index]])
        batch_output = np.array([np.load(filename)[0] for filename in output_filenames[filenames_index]])
        yield batch_input, batch_output
