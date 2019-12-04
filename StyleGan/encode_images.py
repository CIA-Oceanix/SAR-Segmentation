import os
import matplotlib.pyplot as plt
from fire import Fire
import pickle
from tqdm import tqdm
import PIL.Image
import numpy as np

import dnnlib
import dnnlib.tflib as tflib
import config
from encoder.generator_model import Generator
from encoder.perceptual_model import PerceptualModel

from Rignak_Misc.path import get_local_file

SRC_DIR = get_local_file(__file__, os.path.join('datasets', '_invert_src'))
GEN_DIR = get_local_file(__file__, os.path.join('datasets', '_invert_reconstruct'))
DLATENT_DIR = get_local_file(__file__, os.path.join('datasets', '_invert_dlatent'))
LOSS_DIR = get_local_file(__file__, os.path.join('datasets', '_invert_loss'))

MODEL_FILENAME = '2019-02-26-stylegan-faces-network-02048-016041.pkl'
LAYER_NUMBER = {'karras2019stylegan-ffhq-1024x1024.pkl': 18,
                "2019-02-26-stylegan-faces-network-02048-016041.pkl": 16}

BATCH_SIZE = 1
IMAGE_SIZE = 256
LR = 1
ITERATIONS = 5000
ITERATIONS_SAVE = 200
RANDOMIZE_NOISE = False


def split_to_batches(l, n):
    for i in range(0, len(l), n):
        yield l[i:i + n]


def plot_loss(losses, name, average_window=50, loss_dir=LOSS_DIR, first_loss=5):
    losses = losses[first_loss:]  # the first losses are too high and not relevant
    average_loss = np.convolve(losses, np.ones((average_window,)) / average_window, mode='valid')

    plt.figure()
    plt.scatter(range(first_loss, len(losses) + first_loss), losses, marker='.', color='red', s=1)
    plt.plot(range(average_window // 2 + first_loss, len(losses) - average_window // 2 + 1 + first_loss), average_loss)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.yscale('log')
    plt.tight_layout()
    plt.savefig(os.path.join(loss_dir, f'{name}.png'))


def main(src_dir=SRC_DIR, generation_dir=GEN_DIR, dlatent_dir=DLATENT_DIR, batch_size=BATCH_SIZE,
         image_size=IMAGE_SIZE, model_filename=MODEL_FILENAME,
         lr=LR, iterations=ITERATIONS, iteration_save=ITERATIONS_SAVE, randomize_noise=RANDOMIZE_NOISE):
    """
    Find latent representation of reference images using perceptual loss

    :param src_dir: Directory with images for encoding
    :type src_dir: str, optional
    :param generation_dir: Directory for storing generated images
    :type generation_dir: str, optional
    :param generated_dir: Directory for storing dlatent representations
    :type src_dir: str
    :param batch_size: Batch size for generator and perceptual model
    :type batch_size: int, optional
    :param image_size: Size of images for perceptual model
    :type image_size: int, optional
    :param model_filename: Namefile of the model
    :type model_filename: str, optional
    :param lr: Learning rate for perceptual model
    :type lr: float, optional
    :param iterations: Number of optimization steps for each batch
    :type iterations: int, optional
    :param randomize_noise:
    :type randomize_noise: bool, optional
    :return:
    """

    ref_images = [os.path.join(src_dir, x) for x in os.listdir(src_dir)]
    ref_images = list(filter(os.path.isfile, ref_images))

    if len(ref_images) == 0:
        raise Exception(f"{src_dir} is empty")

    os.makedirs(generation_dir, exist_ok=True)
    os.makedirs(dlatent_dir, exist_ok=True)

    # Initialize generator and perceptual model
    tflib.init_tf()
    with open(model_filename, 'rb') as f:
        generator_network, discriminator_network, Gs_network = pickle.load(f)

    generator = Generator(Gs_network, batch_size, randomize_noise=randomize_noise,
                          layer_number=LAYER_NUMBER[model_filename])
    perceptual_model = PerceptualModel(image_size, layer=9, batch_size=batch_size)
    perceptual_model.build_perceptual_model(generator.generated_image)

    # Optimize (only) dlatents by minimizing perceptual loss between reference and generated images in feature space
    for images_batch in tqdm(split_to_batches(ref_images, batch_size), total=len(ref_images) // batch_size):
        names = [os.path.splitext(os.path.basename(x))[0] for x in images_batch]
        new_filename = os.path.join(generation_dir, f'{model_filename}-{names[0]}.png')
        if os.path.exists(new_filename):
            continue

        perceptual_model.set_reference_images(images_batch)
        op = perceptual_model.optimize(generator.dlatent_variable, iterations=iterations, learning_rate=lr)
        pbar = tqdm(op, leave=False, total=iterations)

        losses = []
        best_loss = -1
        for loss in pbar:
            pbar.set_description(f"{' '.join(names)} Loss: {loss}")
            losses.append(loss)
            if not len(losses) % iteration_save:
                # Generate images from found dlatents and save them

                generated_images = generator.generate_images()
                generated_dlatents = generator.get_dlatents()
                for img_array, dlatent, img_name in zip(generated_images, generated_dlatents, names):
                    os.makedirs(os.path.join(generation_dir, img_name), exist_ok=True)
                    os.makedirs(os.path.join(dlatent_dir, img_name), exist_ok=True)

                    img = PIL.Image.fromarray(img_array, 'RGB')
                    img.save(os.path.join(generation_dir, img_name, f'{model_filename}-{img_name}-{len(losses)}.png'))
                    np.save(os.path.join(dlatent_dir, img_name, f'{model_filename}-{img_name}-{len(losses)}.npy'),
                            dlatent)
                    plot_loss(losses, f'{model_filename}-{img_name}.png')

            if best_loss<0 or losses[-1] < best_loss and batch_size == 1:
                best_loss = losses[-1]
                best_img = PIL.Image.fromarray(generator.generate_images()[0], 'RGB')
                best_dlatent = generator.get_dlatents()[0]

        best_img.save(new_filename, 'PNG')
        np.save(os.path.join(dlatent_dir, f'{model_filename}-{img_name}.npy'), best_dlatent)

        print(f"\n{' '.join(names)} Loss: {loss}")

        generator.reset_dlatents()


if __name__ == "__main__":
    Fire(main)
