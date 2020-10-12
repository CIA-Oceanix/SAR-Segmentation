import numpy as np
import cv2

BATCH_SIZE = 8
INPUT_SHAPE = (256, 256, 3)
ZOOM = 0.
ROTATION = 0


def normalize_generator(generator, normalizer, apply_on_output=False):
    while True:
        batch_input, batch_output = next(generator)
        batch_input = normalizer(batch_input)
        if apply_on_output:
            batch_output = normalizer(batch_output)
        yield batch_input, batch_output


def augment_generator(generator, zoom_factor=ZOOM, rotation=ROTATION, noise_function=None, apply_on_output=True):
    while True:
        batch_input, batch_output = next(generator)
        input_shape = batch_input.shape[1:3]
        output_shape = batch_output.shape[1:3]

        angles = (np.random.random(size=batch_input.shape[0]) - 0.5) * rotation
        zooms = 1 + (np.random.random(size=batch_input.shape[0]) - 0.25) * zoom_factor * 2
        h_flips = np.random.randint(0, 2, size=batch_input.shape[0])

        for i, (input_, output, angle, zoom, h_flip) in \
                enumerate(zip(batch_input, batch_output, angles, zooms, h_flips)):
            if h_flip:  # and False:
                input_ = input_[:, ::-1]
                if apply_on_output:
                    output = output[:, ::-1]
            if np.random.random() > 0.8:
                input_ = 1 - input_

            input_rotation_matrix = cv2.getRotationMatrix2D((input_shape[0] // 2, input_shape[1] // 2), angle, zoom)
            if batch_input.shape[-1] != 1:
                batch_input[i] = cv2.warpAffine(input_, input_rotation_matrix, input_shape[:2][::-1])
            else:
                batch_input[i, :, :, 0] = cv2.warpAffine(input_, input_rotation_matrix, input_shape[:2][::-1])

            if apply_on_output and len(output_shape) != 1:  # and False:
                output_rotation_matrix = cv2.getRotationMatrix2D((output_shape[0] // 2, output_shape[1] // 2), angle,
                                                                 zoom)
                if batch_output.shape[1] == 2:
                    if batch_output.shape[-1] == 1:
                        batch_output[i, 1, :, :, 0] = cv2.warpAffine(output, output_rotation_matrix,
                                                                     output_shape[:2][::-1])
                    else:
                        batch_output[i, 1] = cv2.warpAffine(output, output_rotation_matrix, output_shape[:2][::-1])
                else:
                    if batch_output.shape[-1] == 1:
                        batch_output[i, :, :, 0] = cv2.warpAffine(output, output_rotation_matrix,
                                                                  output_shape[:2][::-1])
                    else:
                        batch_output[i] = cv2.warpAffine(output, output_rotation_matrix, output_shape[:2][::-1])

        if noise_function is not None:
            batch_input, batch_output = noise_function(batch_input, batch_output)
        yield batch_input, batch_output


def occlusion_generator(generator, color=(255, 0, 0)):
    next(generator)
    yield None
    while True:
        batch_input, batch_output = next(generator)
        batch_input[:, :, :batch_input.shape[2] // 2] = color
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


def rotsym_augmentor(generator, apply_on_output=True):
    while True:
        batch_input, batch_output = next(generator)
        symmetries = np.random.randint(0, 2, size=(batch_input.shape[0], 2)) * 2 - 1
        rotations = np.random.randint(0, 4, size=(batch_input.shape[0]))
        for i, ((vertical_symmetry, horizontal_symmetry), rotation) in enumerate(zip(symmetries, rotations)):
            batch_input[i] = np.rot90(batch_input[i, ::vertical_symmetry, ::horizontal_symmetry], k=rotation)
            if apply_on_output:
                batch_output[i] = np.rot90(batch_output[i, ::vertical_symmetry, ::horizontal_symmetry], k=rotation)
        yield batch_input, batch_output
