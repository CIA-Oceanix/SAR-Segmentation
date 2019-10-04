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

from Rignak_Misc.path import get_local_file

DEFAULT_MODEL = 'https://drive.google.com/uc?id=1MEGjdvVpUsu1jB4zrXZN7Y4kBBOzizDQ'
RESULT_ROOT = get_local_file(__file__, 'results')
TRUNCATION_PSI = 0.7
NUMBER_OF_PICTURES = 20


def main(model_filename=DEFAULT_MODEL, truncation_psi=TRUNCATION_PSI, result_root=RESULT_ROOT,
         number_of_pictures=NUMBER_OF_PICTURES):
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

    for i in range(number_of_pictures):
        latents = np.random.randn(1, Gs.input_shape[1])

        # Generate image.
        fmt = dict(func=tflib.convert_images_to_uint8, nchw_to_nhwc=True)
        images = Gs.run(latents, None, truncation_psi=truncation_psi, randomize_noise=True, output_transform=fmt)

        # Save image.
        png_filename = os.path.join(result_root, os.path.splitext(os.path.split(model_filename)[-1])[0] + f'{i}.png')
        PIL.Image.fromarray(images[0], 'RGB').save(png_filename)


if __name__ == "__main__":
    fire.Fire(main)