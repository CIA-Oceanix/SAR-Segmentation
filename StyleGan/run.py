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

from Rignak_Misc.path import get_local_file

DEFAULT_MODEL = '2019-02-26-stylegan-faces-network-02048-016041.pkl'
LAYER_NUMBER = 16
RESULT_ROOT = get_local_file(__file__, 'results')
TRUNCATION_PSI = 0.7
NUMBER_OF_PICTURES = 20


def get_generative(model, truncation_psi=TRUNCATION_PSI):
    def generative(latents):
        if len(latents.shape) == 1:
            latents = np.expand_dims(latents, 0)
        return model.run(latents, None, randomize_noise=False, truncation_psi=truncation_psi, use_noise=False, output_transform=fmt)

    fmt = dict(func=tflib.convert_images_to_uint8, nchw_to_nhwc=True)
    return generative


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
    generative = get_generative(Gs, truncation_psi=truncation_psi)

    layers = {name:tensor for name, tensor, _ in Gs.list_layers()}
    for i in tqdm.trange(number_of_pictures):
        latents = np.random.randn(1, Gs.input_shape[1])

        # Generate image.
        images = generative(latents)
        truncation_output = tflib.run(layers['Truncation'], feed_dict={layers['latents_in']:latents})

        # Save image.
        filename = os.path.join(result_root, os.path.splitext(os.path.split(model_filename)[-1])[0] + f'{i}')
        PIL.Image.fromarray(images[0], 'RGB').save(f"{filename}.png")
        np.save(f'{filename}_1.npy', latents[0])
        np.save(f'{filename}_16.npy', truncation_output[0])


if __name__ == "__main__":
    fire.Fire(main)
