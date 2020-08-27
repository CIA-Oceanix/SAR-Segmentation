import os, sys
import numpy as np
import tqdm
import time

if __name__ == '__main__':
    sys.path.append(os.path.join(os.path.dirname(__file__), "..", '..'))
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

from keras.applications.inception_v3 import InceptionV3
from keras.models import Model, load_model
from keras.layers import Input, Dense, GlobalAveragePooling2D, Conv2D, concatenate
from keras_radam.training import RAdamOptimizer

from keras import backend as K

from Rignak_Misc.path import get_local_file
from Rignak_DeepLearning.BiOutput import generator
from Rignak_DeepLearning.Autoencoders.plot_example import plot_example
from Rignak_DeepLearning.data import get_dataset_roots
from Rignak_DeepLearning.Autoencoders import unet
from Rignak_DeepLearning.generator import autoencoder_generator, occlusion_generator

import matplotlib.pyplot as plt


def mse(y_true, y_pred):
    return K.mean(K.square(y_pred[:, :, y_pred.shape[2] // 2:] - y_true[:, :, y_pred.shape[2] // 2:]), axis=-1)


class UGAN():
    def __init__(self, image_shape):
        self.img_rows, self.img_cols, self.channels = image_shape
        self.img_shape = image_shape

        optimizer = RAdamOptimizer(1e-2)

        patch = int(self.img_rows / 2 ** 4)
        self.disc_patch = (patch, patch, 1)
        self.disc_patch = (6, 6, 1)

        self.DISCRIMINATOR_FILENAME = os.path.join(ROOT, 'discriminator.h5')
        self.ENCODER_FILENAME = os.path.join(ROOT, 'ugan_unet.h5')

        if os.path.exists(self.DISCRIMINATOR_FILENAME) and False:
            self.discriminator = load_model(self.DISCRIMINATOR_FILENAME)
        else:
            self.discriminator = build_discriminator(img_shape=self.img_shape)
        self.discriminator.compile(loss='mse', optimizer=optimizer, metrics=['accuracy'])

        if os.path.exists(self.ENCODER_FILENAME) and False:
            self.unet = load_model(self.ENCODER_FILENAME)
        else:
            config = {'INPUT_SHAPE': self.img_shape,
                      "ACTIVATION": "sigmoid",
                      "CONV_LAYERS": [16, 32, 64, 128]}
            self.unet = unet.import_model(config=config)

        img = Input(shape=self.img_shape)
        reconstructed_img = self.unet(img)

        self.discriminator.trainable = False
        validity = self.discriminator(reconstructed_img)

        self.adversarial_autoencoder = Model(img, [reconstructed_img, validity], name="adversarial")
        self.adversarial_autoencoder.compile(loss=[mse, 'mse'], loss_weights=[1, 0.01], optimizer=optimizer)

        self.discriminator.summary()
        self.unet.summary()
        self.adversarial_autoencoder.summary()

    def train(self, dataset, epochs, batch_size=16, steps_per_epoch=200):
        def process_batch(generator):
            batch_input, batch_output = next(generator)

            reconstructed_batch = self.unet.predict(batch_input)

            # Train the discriminator
            d_loss = self.discriminator.train_on_batch(batch_input, valid)
            d_loss += self.discriminator.train_on_batch(reconstructed_batch, fake)

            # Train the generator
            generator_loss = self.adversarial_autoencoder.train_on_batch(batch_input, [batch_output, valid])
            return generator_loss, d_loss

        # Adversarial loss ground truths
        valid = np.ones((batch_size,) + self.disc_patch)
        valid[:, :, 3:] = 0
        fake = np.zeros((batch_size,) + self.disc_patch)

        train_folder, val_folder = get_dataset_roots('', dataset=dataset)

        train_generator = autoencoder_generator(train_folder, input_shape=self.img_shape, batch_size=batch_size)
        train_generator = occlusion_generator(train_generator, color=0)
        next(train_generator)

        val_generator = autoencoder_generator(val_folder, input_shape=self.img_shape, batch_size=batch_size)
        val_generator = occlusion_generator(val_generator, color=0)
        next(val_generator)

        for epoch in range(epochs):
            pbar = tqdm.trange(steps_per_epoch)

            unet_loss = []
            discriminative_loss = []
            discriminative_acc = []
            for batch_i in range(steps_per_epoch):
                generator_loss, d_loss = process_batch(train_generator)
                unet_loss.append(generator_loss[0])
                discriminative_loss.append(generator_loss[2])
                discriminative_acc.append(d_loss[1] * 100)
                pbar.set_description(f"Epoch {epoch}/{epochs} - Batch {batch_i + 1}/{steps_per_epoch} - "
                                     f"[D_loss = {np.mean(discriminative_loss):.4f}"
                                     f" D_acc = {np.mean(discriminative_acc):.1f}%"
                                     f" E_loss = {np.mean(unet_loss):.4f}"
                                     )
                pbar.update(1)

            self.sample_images(dataset, val_generator, epoch)
            if epoch == 0:
                print(f"Asc Time;Epoch;Discriminator loss;Discriminator accuracy;"
                      f"Encoder loss;Encoder class accuracy;Decoder loss\n")
            print(f"{time.asctime()};{epoch};{np.mean(discriminative_loss)};{np.mean(discriminative_acc)};"
                  f"{np.mean(unet_loss)}\n")

            self.discriminator.save(self.DISCRIMINATOR_FILENAME)
            self.unet.save(self.ENCODER_FILENAME)

    def sample_images(self, dataset, generator, epoch):
        os.makedirs(ROOT, exist_ok=True)
        batch_input, batch_output = next(generator)
        reconstruction = self.unet.predict(batch_input)

        plot_example(batch_input, reconstruction, batch_output)
        plt.savefig(os.path.join(ROOT, f"{epoch}.png"))
        plt.savefig(os.path.join(ROOT, f"_.png"))
        plt.close()


def build_discriminator(img_shape):
    if img_shape[-1] == 1:
        img_input = Input(shape=img_shape)
        img_conc = concatenate([img_input, img_input, img_input])
        base_model = InceptionV3(input_tensor=img_conc, classes=1, include_top=False)
    else:
        base_model = InceptionV3(input_shape=img_shape, classes=1, include_top=False)
    x = base_model.output
    x = Conv2D(1, (1, 1), activation='sigmoid')(x)
    # x = GlobalAveragePooling2D()(x)
    # x = Dense(1, activation='sigmoid')(x)
    model = Model(inputs=base_model.input, outputs=x, name="discriminator")
    return model


if __name__ == '__main__':
    # dataset = sys.argv[1]
    dataset = "waifu"
    ROOT = f"_output/UGAN_{dataset}"
    gan = UGAN((256, 256, 1))
    gan.train(dataset, epochs=50, batch_size=4, steps_per_epoch=1000)
