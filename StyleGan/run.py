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

from skimage import transform

from Rignak_Misc.path import get_local_file

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

DEFAULT_MODEL = 'sar_network-snapshot-008100.pkl'
LAYER_NUMBER = 18
RESULT_ROOT = get_local_file(__file__, 'results')
TRUNCATION_PSI = 0.7
NUMBER_OF_PICTURES = 20
THUMB_SIZE = 512


def get_generative(model, truncation_psi=TRUNCATION_PSI):
    def generative(latents, noise=False):
        if len(latents.shape) == 1:
            latents = np.expand_dims(latents, 0)
        return model.run(latents, None, randomize_noise=noise, truncation_psi=truncation_psi, use_noise=noise,
                         output_transform=fmt)

    fmt = dict(func=tflib.convert_images_to_uint8, nchw_to_nhwc=True)
    return generative


def main(model_filename=DEFAULT_MODEL, truncation_psi=TRUNCATION_PSI, result_root=RESULT_ROOT,
         number_of_pictures=NUMBER_OF_PICTURES, thumb_size=THUMB_SIZE):
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

    layers = {name: tensor for name, tensor, _ in Gs.list_layers()}
    for i in tqdm.trange(number_of_pictures):
        latents = np.random.randn(1, Gs.input_shape[1])

        # Generate image.
        images = generative(latents)
        truncation_output = tflib.run(layers['Truncation'], feed_dict={layers['latents_in']: latents})

        for j, images in enumerate([generative(latents, noise=False), generative(latents, noise=True),
                                    generative(latents, noise=True), generative(latents, noise=True)]):
            # Save image.
            filename = os.path.join(result_root, f"{os.path.splitext(os.path.split(model_filename)[-1])[0]}_{i}_{j}")
            if images.shape[-1] == 1:
                image = transform.resize(images[0, :, :, 0], (thumb_size, thumb_size)) * 255
                im = PIL.Image.fromarray(image).convert("L")
                im.save(f"{filename}.png")
            else:
                image = PIL.Image.fromarray(images[0], 'RGB')
                image.resize((thumb_size, thumb_size), PIL.Image.BICUBIC).save(f"{filename}.png")
            np.save(f'{filename}.npy', truncation_output[0][0])

        # check noise influence
        amount4std = 100
        variations = np.zeros((images.shape[1], images.shape[2], amount4std))
        for j in range(amount4std):
            variations[:, :, j] = np.mean(generative(latents, noise=True), axis=-1)
        variations = np.std(variations, axis=-1)
        variations = (variations - variations.min())/(variations.max()-variations.min())*255
        filename = os.path.join(result_root, f"{os.path.splitext(os.path.split(model_filename)[-1])[0]}_{i}_std.png")
        PIL.Image.fromarray(variations).convert("L").save(filename)


if __name__ == "__main__":
    fire.Fire(main)
