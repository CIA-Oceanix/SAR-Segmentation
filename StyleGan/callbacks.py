import os
import matplotlib.pyplot as plt
import numpy as np

from keras.callbacks import Callback

from Rignak_DeepLearning.StyleGan.encoder.generator_model import Generator
from Rignak_DeepLearning.StyleGan.load import load_model
from Rignak_Misc.plt import imshow

EXAMPLE_CALLBACK_ROOT = os.path.join('_outputs', 'example')


class GanRegressorExampleCallback(Callback):
    def __init__(self, generator, gan_filename, layer_number, truncation_psi, root=EXAMPLE_CALLBACK_ROOT,
                 denormalizer=None):
        super().__init__()
        self.root = root
        self.generator = generator
        self.layer_number = layer_number
        Gs_network = load_model(gan_filename)
        self.generate = Generator(Gs_network,
                                  batch_size=1,
                                  layer_number=layer_number,
                                  truncation_psi=truncation_psi).generate_images
        self.denormalizer = denormalizer

    def on_train_begin(self, logs=None):
        os.makedirs(os.path.join(self.root, self.model.name), exist_ok=True)
        self.on_epoch_end(0, logs=logs)

    def on_epoch_end(self, epoch, logs=None):
        example = next(self.generator)
        plot_example(example[0][:8], self.model.predict(example[0][:8]), self.generate, layer_number=self.layer_number,
                     denormalizer=self.denormalizer)
        plt.savefig(os.path.join(self.root, self.model.name, f'{self.model.name}_{epoch}.png'))
        plt.savefig(os.path.join(self.root, f'{self.model.name}_current.png'))
        plt.close()


def plot_example(input_images, predictions, generate, layer_number=1, denormalizer=None):
    n = input_images.shape[0]

    plt.figure(figsize=(20, 10))
    for i, (input_image, prediction) in enumerate(zip(input_images, predictions)):
        if layer_number != 1:
            prediction = np.vstack([prediction] * layer_number)

        plt.subplot(2, n, 1 + i)
        imshow(input_image, denormalizer=denormalizer)
        if not i:
            plt.title("Original")

        plt.subplot(2, n, 1 + i + n)
        imshow(generate(prediction)[0], denormalizer=denormalizer)
        if not i:
            plt.title("Reconstruction")

    plt.tight_layout()
