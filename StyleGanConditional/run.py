# Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the Creative Commons Attribution-NonCommercial
# 4.0 International License. To view a copy of this license, visit
# http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to
# Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.

"""Minimal script for generating an image using pre-trained StyleGAN generator."""

import os
import pickle
import PIL.Image
import fire
import dnnlib.tflib as tflib
import numpy as np
import tqdm
import tensorflow as tf
import skimage
import skimage.transform

from Rignak_Misc.path import get_local_file

DEFAULT_MODEL = 'network-snapshot-007750.pkl'
LAYER_NUMBER = 12
RESULT_ROOT = get_local_file(__file__, 'results')
TRUNCATION_PSI = 1.0
THUMB_SIZE = 128

### Add labels
LABELS = [(label, np.eye(8)[i])
          for i, label in enumerate(("Blue Mountain", "Chino", "Chiya", "Cocoa", "Maya", "Megumi", "Rize", "Sharo"))]
LABELS = [(label, np.eye(10)[i])
          for i, label in enumerate(("Atm Front", "Bio Slick", "Iceberg", "Low Wind", "Conv Cells", "Oceanic Front",
                                     "Ocean Waves", "Rain Cells", "Sea Ice", "Wind Streaks"))]

# Add even combination labels
# LABELS.append((f"{LABELS[2][0]}+{LABELS[6][0]}", LABELS[2][1] + LABELS[6][1]))
# LABELS.append((f"0.5x{LABELS[2][0]}+0.5x{LABELS[6][0]}", 0.5 * LABELS[2][1] + 0.5 * LABELS[6][1]))
# LABELS.append((f"{LABELS[2][0]}+{LABELS[6][0]}", LABELS[2][1] + LABELS[6][1]))
#
# # Add uneven combinations
# for a, b in [(0.0, 1.0), (0.25, 0.75), (0.5, 0.5), (0.75, 0.25)]:
#     LABELS.append((f"{a}x{LABELS[2][0]}+{b}x{LABELS[6][0]}", a * LABELS[2][1] + b * LABELS[6][1]))

# Add ALL and NONE
# LABELS.append((f"None", np.zeros(10)))
# LABELS.append((f"All", np.ones(10)))

NUMBER_OF_PICTURES = 4 * len(LABELS)


def get_generative(model, truncation_psi=TRUNCATION_PSI):
    def generative(latents, label_input=None):
        if len(latents.shape) == 1:
            latents = np.expand_dims(latents, 0)
        return model.run(latents, label_input, randomize_noise=False, truncation_psi=truncation_psi, use_noise=True,
                         output_transform=fmt)

    fmt = dict(func=tflib.convert_images_to_uint8, nchw_to_nhwc=True)
    return generative


def main(model_filename=DEFAULT_MODEL, truncation_psi=TRUNCATION_PSI, result_root=RESULT_ROOT,
         number_of_pictures=NUMBER_OF_PICTURES, thumb_size=THUMB_SIZE, labels=LABELS):
    """
    Create images from a given model

    :param model_filename: name of the model to use
    :param truncation_psi: originality factor, close to 0 means less originality
    :param result_root: name of the folder to contain the output
    :param number_of_pictures: number of picture to generate
    :return:
    """
    os.makedirs(result_root, exist_ok=True)

    # Initialize TensorFlow.
    tflib.init_tf()

    # Load pre-trained network.
    with open(model_filename, 'rb') as f:
        _, _, Gs = pickle.load(f)
        for layer in Gs.list_layers():
            print(layer[0])
    generative = get_generative(Gs, truncation_psi=truncation_psi)

    layers = {name: tensor for name, tensor, _ in Gs.list_layers()}
    mosaic = np.zeros((NUMBER_OF_PICTURES // len(labels) * THUMB_SIZE, len(labels) * THUMB_SIZE, 3))
    for i in tqdm.trange(number_of_pictures):
        label_name, label_input = labels[i % len(labels)]
        label_input = np.expand_dims(label_input, axis=0)
        label_input = label_input.astype(float)
        latents = np.random.randn(1, Gs.input_shape[1])

        # Generate image.
        images = generative(latents, label_input=label_input)
        truncation_output = tflib.run(layers['Truncation'], feed_dict={layers['latents_in']: latents,
                                                                       layers['labels_in']: label_input})

        # Save image.
        filename = os.path.join(result_root,
                                f"{os.path.splitext(os.path.split(model_filename)[-1])[0]}_{label_name}_{i}")
        if images.shape[-1] == 1:
            image = skimage.transform.resize(images[0, :, :, 0], (thumb_size, thumb_size))
            image = (np.expand_dims(image, axis=-1)[:,:,[0,0,0]]*255).astype('uint8')
            image = PIL.Image.fromarray(image)
        else:
            image = PIL.Image.fromarray(images[0], 'RGB')
        image = image.resize((thumb_size, thumb_size), PIL.Image.BICUBIC)
        np.save(f'{filename}.npy', truncation_output[0])
        image.save(f"{filename}.png")

        mosaic[(i // len(labels)) * THUMB_SIZE:(i // len(labels) + 1) * THUMB_SIZE,
        (i % len(labels)) * THUMB_SIZE:(i % len(labels) + 1) * THUMB_SIZE] = image

    print(mosaic.max())
    mosaic = PIL.Image.fromarray(mosaic.astype('uint8'))
    mosaic.save(os.path.join(result_root, f"{os.path.splitext(os.path.split(model_filename)[-1])[0]}_mosaic.png"))

if __name__ == "__main__":
    fire.Fire(main)
