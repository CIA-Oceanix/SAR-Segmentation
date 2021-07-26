import numpy as np
import os
import glob
from skimage.transform import resize
import cv2

from Rignak_DeepLearning.generator import get_add_additional_inputs
from Rignak_DeepLearning.data import read
from Rignak_Misc.path import list_dir, convert_link

BATCH_SIZE = 8
INPUT_SHAPE = (512, 512, 3)
OUTPUT_SHAPE = (32, 32, 1)
INPUT_LABEL = "input"
OUTPUT_LABEL = "output"


def image_distance_base_generator(root, batch_size=BATCH_SIZE, input_shape=INPUT_SHAPE):
    labels = os.listdir(root)
    filenames = np.array([np.array(glob.glob(f"{root}/{label}/*.png")) for label in labels])
    label_indexes = [i for i, label_filenames in enumerate(filenames) if len(label_filenames)]
    k = 0
    n_files = len(filenames)
    yield None
    while True:
        output = np.zeros((batch_size, 3))
        output[:, 0] = np.random.choice(label_indexes, size=batch_size)
        output[:batch_size // 2, 1] = [np.random.choice([i2 for i2 in label_indexes if i2 != i1]) for i1 in
                                       output[:batch_size // 2, 0]]
        output[batch_size // 2:, 1] = output[batch_size // 2:, 0]
        output[:, 2] = np.equal(output[:, 0], output[:, 1], dtype=float)

        batch_path_1 = [np.random.choice(filenames[int(i)]) for i in output[:, 0]]
        batch_path_2 = [np.random.choice(filenames[int(i)]) for i in output[:, 1]]

        batch_input_1 = np.array([read(path, input_shape) for path in batch_path_1])
        batch_input_2 = np.array([read(path, input_shape) for path in batch_path_2])
        yield [batch_input_1, batch_input_2], output


def siamese_augment_generator(generator, zoom_factor=0, rotation=0, border_value=(1, 1, 1)):
    while True:
        (batch_input_1, batch_input_2), batch_output = next(generator)
        if not zoom_factor and not rotation:
            yield (batch_input_1, batch_input_2), batch_output
            continue

        batch_image_input_1 = batch_input_1[0] if isinstance(batch_input_1, list) else batch_input_1
        batch_image_input_2 = batch_input_2[0] if isinstance(batch_input_2, list) else batch_input_2
        input_shape = batch_image_input_1.shape[1:3]

        angles = (np.random.random(size=batch_image_input_1.shape[0]) - 0.5) * rotation
        zooms = 1 + (np.random.random(size=batch_image_input_1.shape[0]) - 0.25) * zoom_factor * 2
        h_flips = np.random.randint(0, 2, size=batch_image_input_1.shape[0])

        for i, (input_1, input_2, output, angle, zoom, h_flip) in \
                enumerate(zip(batch_image_input_1, batch_image_input_2, batch_output, angles, zooms, h_flips)):
            if h_flip:
                input_1 = input_1[:, ::-1]
                input_2 = input_2[:, ::-1]

            input_rotation_matrix = cv2.getRotationMatrix2D((input_shape[1] // 2, input_shape[0] // 2), angle, zoom)
            if batch_image_input_1.shape[-1] != 1:
                batch_image_input_1[i] = cv2.warpAffine(input_1, input_rotation_matrix, input_shape[:2][::-1],
                                                        borderValue=border_value)
                batch_image_input_2[i] = cv2.warpAffine(input_2, input_rotation_matrix, input_shape[:2][::-1],
                                                        borderValue=border_value)
            else:
                batch_image_input_1[i, :, :, 0] = cv2.warpAffine(input_1, input_rotation_matrix, input_shape[:2][::-1],
                                                                 borderValue=border_value)
                batch_image_input_2[i, :, :, 0] = cv2.warpAffine(input_2, input_rotation_matrix, input_shape[:2][::-1],
                                                                 borderValue=border_value)

        batch_input = [batch_image_input_1, batch_image_input_2]
        yield batch_input, batch_output
