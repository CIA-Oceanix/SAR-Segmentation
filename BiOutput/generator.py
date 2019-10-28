import numpy as np
import os
import cv2

from Rignak_DeepLearning.data import read

BATCH_SIZE = 8
INPUT_SHAPE = (256, 256, 3)
ZOOM = 0.2
ROTATION = 20


def make_categorizer_output(index, label_number):
    output = np.zeros((label_number))
    output[index] = 1
    return output


def generator(root, batch_size=BATCH_SIZE, input_shape=INPUT_SHAPE):
    folders = os.listdir(root)
    tags = {os.path.join(root, folder, filename): make_categorizer_output(folders.index(folder), len(folders))
            for folder in folders
            for filename in os.listdir(os.path.join(root, folder))}
    filenames = list(tags.keys())

    while True:
        batch_path = np.random.choice(filenames, size=batch_size)

        batch_input = np.array([read(path, input_shape=input_shape) for path in batch_path])
        batch_image_output = batch_input.copy()
        batch_label_output = np.array([tags[filename] for filename in batch_path])
        batch_output = [batch_label_output, batch_image_output]
        yield batch_input, batch_output


def normalize_generator(generator, normalization_function, apply_on_output=False):
    while True:
        batch_input, batch_output = next(generator)
        batch_input = normalization_function(batch_input)
        if apply_on_output:
            batch_output[1] = normalization_function(batch_output[1])

        yield batch_input, batch_output


def augment_generator(generator, zoom_factor=ZOOM, rotation=ROTATION, noise_function=None, apply_on_output=False):
    while True:
        batch_input, batch_output = next(generator)
        input_shape = batch_input.shape[1:3]

        angles = (np.random.random(size=batch_input.shape[0]) - 0.5) * rotation
        zooms = np.random.random(size=batch_input.shape[0]) * zoom_factor + 1
        h_flips = np.random.randint(0, 2, size=batch_input.shape[0])

        for i, (input_, output, angle, zoom, h_flip) in \
                enumerate(zip(batch_input, batch_output[1], angles, zooms, h_flips)):
            if h_flip:
                input_ = input_[:, ::-1]
                if apply_on_output:
                    output = output[:, ::-1]

            rotation_matrix = cv2.getRotationMatrix2D((input_shape[0] // 2, input_shape[1] // 2), angle, zoom)
            batch_input[i] = cv2.warpAffine(input_, rotation_matrix, input_shape[:2])
            if apply_on_output:
                if output.shape[-1] == 1:
                    batch_output[1][i, :, :, 0] = cv2.warpAffine(output, rotation_matrix, input_shape[:2])
                else:
                    batch_output[1][i] = cv2.warpAffine(output, rotation_matrix, input_shape[:2])

        if noise_function is not None:
            batch_input = noise_function(batch_input)

        yield batch_input, batch_output
