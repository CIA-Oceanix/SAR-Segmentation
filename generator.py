import numpy as np
import cv2
import os
import json

BATCH_SIZE = 8
INPUT_SHAPE = (256, 256, 3)
ZOOM = 0.
ROTATION = 0


def augment_generator(generator, zoom_factor=ZOOM, rotation=ROTATION, noise_function=None, apply_on_output=True, border_value=(1,1,1)):
    while True:
        batch_input, batch_output = next(generator)
        if not zoom_factor and not rotation:
            yield batch_input, batch_output
            continue
        
        batch_image_input = batch_input[0] if isinstance(batch_input, list) else batch_input
        input_shape = batch_image_input.shape[1:3]
        output_shape = batch_output.shape[1:3]

        angles = (np.random.random(size=batch_image_input.shape[0]) - 0.5) * rotation
        zooms = 1 + (np.random.random(size=batch_image_input.shape[0]) - 0.25) * zoom_factor * 2
        h_flips = np.random.randint(0, 2, size=batch_image_input.shape[0])
        
        for i, (input_, output, angle, zoom, h_flip) in \
                enumerate(zip(batch_image_input, batch_output, angles, zooms, h_flips)):
            if h_flip and False:
                input_ = input_[:, ::-1]
                if apply_on_output:
                    output = output[:, ::-1]

            input_rotation_matrix = cv2.getRotationMatrix2D((input_shape[1] // 2, input_shape[0] // 2), angle, zoom)
            if batch_image_input.shape[-1] != 1:
                batch_image_input[i] = cv2.warpAffine(input_, input_rotation_matrix, input_shape[:2][::-1], borderValue=border_value)
            else:
                batch_image_input[i, :, :, 0] = cv2.warpAffine(input_, input_rotation_matrix, input_shape[:2][::-1], borderValue=border_value)
            
            
            if apply_on_output and len(output_shape) != 1:
                output_rotation_matrix = cv2.getRotationMatrix2D((output_shape[0] // 2, output_shape[1] // 2), angle,
                                                                 zoom)
                if batch_output.shape[1] == 2:
                    if batch_output.shape[-1] == 1:
                        batch_output[i, 1, :, :, 0] = cv2.warpAffine(output, output_rotation_matrix,
                                                                     output_shape[:2][::-1], borderValue=0)
                    else:
                        batch_output[i, 1] = cv2.warpAffine(output, output_rotation_matrix, output_shape[:2][::-1], borderValue=0)
                else:
                    if batch_output.shape[-1] == 1:
                        batch_output[i, :, :, 0] = cv2.warpAffine(output, output_rotation_matrix,
                                                                  output_shape[:2][::-1], borderValue=0)
                    else:
                        batch_output[i] = cv2.warpAffine(output, output_rotation_matrix, output_shape[:2][::-1], borderValue=0)
        if noise_function is not None:
            batch_input, batch_output = noise_function(batch_input, batch_output)
        batch_input = [batch_image_input, batch_input[1]] if isinstance(batch_input, list) else batch_image_input
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
        batch_image_input = batch_input[0] if isinstance(batch_input, list) else batch_input
        symmetries = np.random.randint(0, 2, size=(batch_image_input.shape[0], 2)) * 2 - 1
        rotations = np.random.randint(0, 4, size=(batch_image_input.shape[0]))
        for i, ((vertical_symmetry, horizontal_symmetry), rotation) in enumerate(zip(symmetries, rotations)):
            batch_image_input[i] = np.rot90(batch_image_input[i, ::vertical_symmetry, ::horizontal_symmetry],
                                            k=rotation)
            if apply_on_output:
                batch_output[i] = np.rot90(batch_output[i, ::vertical_symmetry, ::horizontal_symmetry], k=rotation)
        batch_input = [batch_image_input, batch_input[1]] if isinstance(batch_input, list) else batch_image_input
        yield batch_input, batch_output


def get_add_additional_inputs(root, attributes):
    print('attributes:', attributes)
    if not attributes:
        return lambda batch_input, batch_input_path: batch_input
    normalized_inputs, _ = get_normalized_inputs(os.path.split(root)[0], attributes)

    def add_additional_inputs(batch_input, batch_input_path):
        additional_inputs_array = np.zeros((batch_input.shape[0], len(attributes)))
        for i_filename, full_filename in enumerate(batch_input_path):
            filename = os.path.split(full_filename)[-1]
            additional_inputs_array[i_filename] = normalized_inputs.get(filename)
        return [batch_input, additional_inputs_array]

    return add_additional_inputs


def get_normalized_inputs(root, attributes, input_filename="output.json", normalization_filename="normalization.json"):
    with open(os.path.join(root, input_filename), 'r') as json_file:
        additional_inputs = json.load(json_file)
    with open(os.path.join(root, normalization_filename), 'r') as json_file:
        normalization_dict = json.load(json_file)

    normalized_inputs = {}
    for attribute_index, attribute_name in enumerate(attributes):
        mean = normalization_dict[attribute_name]['mean']
        std = normalization_dict[attribute_name]['std']
        for filename, entry in additional_inputs.items():
            if filename not in normalized_inputs:
                normalized_inputs[filename] = np.zeros(len(attributes))

            value = entry.get(attribute_name)
            if np.isscalar(value) and -10 ** 10 < value < 10 ** 10:  # check if NaN
                normalized_inputs[filename][attribute_index] = (value - mean) / std
            else:
                normalized_inputs[filename][attribute_index] = 0

    normalized_inputs = {key: value for key, value in normalized_inputs.items()}
    return normalized_inputs, (normalization_dict, additional_inputs)
