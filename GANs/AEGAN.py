import os, sys
import numpy as np
import tqdm
import time

if __name__ == '__main__':
    sys.path.append(os.path.join(os.path.dirname(__file__), "..", '..'))
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

from keras.applications.inception_v3 import InceptionV3
from keras.models import Model, load_model
from keras.layers import Input, Dense, GlobalAveragePooling2D, concatenate
from keras_radam.training import RAdamOptimizer

from Rignak_DeepLearning.normalization import NORMALIZATION_FUNCTIONS
from Rignak_Misc.path import get_local_file
from Rignak_DeepLearning.BiOutput import generator, callbacks
from Rignak_DeepLearning.data import get_dataset_roots
from Rignak_DeepLearning.BiOutput.multiscale_autoencoder import build_decoder, build_encoder

from Rignak_DeepLearning.GANs.pix2pix import build_discriminator as build_pix2pix_discriminator

import matplotlib.pyplot as plt

LOG_FILENAME = get_local_file(__file__, "AEGAN_LOGS.csv")


class AEGAN():
    def __init__(self, image_shape, n_classes):
        labels = [''] * n_classes
        self.img_rows, self.img_cols, self.channels = image_shape
        self.img_shape = image_shape

        self.latent_space_shapes = ([8, 8, 2], [8, 8, 4], [8, 8, 8], [8, 8, 16])
        self.n_classes = n_classes
        self.convs = [16, 32, 64, 128]
        optimizer = RAdamOptimizer(1e-3)

        patch = int(self.img_rows / 2 ** 4)
        self.disc_patch = (patch, patch, 1)

        self.DISCRIMINATOR_FILENAME = os.path.join(ROOT, 'discriminator.h5')
        self.ENCODER_FILENAME = os.path.join(ROOT, 'encoder.h5')
        self.DECODER_FILENAME = os.path.join(ROOT, 'decoder.h5')

        if os.path.exists(self.DISCRIMINATOR_FILENAME):
            self.discriminator = load_model(self.DISCRIMINATOR_FILENAME)
        else:
            self.discriminator = build_pix2pix_discriminator(img_shape=self.img_shape, df=64, inputs=1)
        self.discriminator.compile(loss='mse', optimizer=optimizer, metrics=['accuracy'])

        if os.path.exists(self.ENCODER_FILENAME):
            self.encoder = load_model(self.ENCODER_FILENAME)
        else:
            self.encoder = build_encoder(input_shape=self.img_shape, latent_space_shapes=self.latent_space_shapes,
                                         activation='relu', labels=labels, conv_layers=self.convs)

        if os.path.exists(self.DECODER_FILENAME):
            self.decoder = load_model(self.DECODER_FILENAME)
        else:
            self.decoder = build_decoder(input_shape=self.img_shape, latent_space_shapes=self.latent_space_shapes,
                                         labels=labels, conv_layers=self.convs, activation='relu')

        img = Input(shape=self.img_shape)
        encoded_repr = self.encoder(img)
        reconstructed_img = self.decoder(encoded_repr)

        self.discriminator.trainable = False
        validity = self.discriminator(reconstructed_img)

        self.adversarial_autoencoder = Model(img, [encoded_repr[-1], reconstructed_img, validity], name="adversarial")
        self.adversarial_autoencoder.compile(loss={'encoder': 'categorical_crossentropy',
                                                   'decoder': 'mae',
                                                   'discriminator': 'mse'},
                                             loss_weights={"encoder": 1, "decoder": 10, "discriminator": 0.1},
                                             optimizer=optimizer)
        self.encoder.summary()
        self.decoder.summary()
        self.adversarial_autoencoder.summary()

    def train(self, dataset, epochs, batch_size=16, steps_per_epoch=200,
              log_filename=LOG_FILENAME):
        def process_batch(generator):
            batch_input, (batch_label, batch_output) = next(generator)

            latent_batch = self.encoder.predict(batch_input)
            reconstructed_batch = self.decoder.predict(latent_batch)

            predicted_labels = latent_batch[-1]
            correct_classification_rate = np.mean([int(np.argmax(pred) == np.argmax(truth))
                                                   for pred, truth in zip(batch_label, predicted_labels)])

            # Train the discriminator
            d_loss = self.discriminator.train_on_batch(batch_input, valid)
            d_loss += self.discriminator.train_on_batch(reconstructed_batch, fake)

            # Train the generator
            generator_loss = self.adversarial_autoencoder.train_on_batch(batch_input,
                                                                         [batch_label, batch_output, valid])
            return generator_loss, correct_classification_rate, d_loss

        # Adversarial loss ground truths
        valid = np.ones((batch_size,) + self.disc_patch)
        fake = np.zeros((batch_size,) + self.disc_patch)

        train_folder, val_folder = get_dataset_roots('', dataset=dataset)
        normalizer, denormalizer = NORMALIZATION_FUNCTIONS['fourier']()  # TODO as argv

        labels = [folder for folder in os.listdir(train_folder)
                  if os.path.isdir(os.path.join(train_folder, folder))]
        train_generator = generator.generator(train_folder, batch_size=batch_size, input_shape=self.img_shape)
        train_generator = generator.normalize_generator(train_generator, normalizer, apply_on_output=True)

        val_generator = generator.generator(val_folder, batch_size=batch_size, input_shape=self.img_shape)
        val_generator = generator.normalize_generator(val_generator, normalizer, apply_on_output=True)

        for epoch in range(epochs):
            pbar = tqdm.trange(steps_per_epoch)

            encoder_loss = []
            encoder_acc = []
            decoder_loss = []
            discriminative_loss = []
            discriminative_acc = []
            for batch_i in range(steps_per_epoch):
                generator_loss, correct_classification_rate, d_loss = process_batch(train_generator)
                encoder_loss.append(generator_loss[0])
                encoder_acc.append(correct_classification_rate * 100)
                decoder_loss.append(generator_loss[1])
                discriminative_loss.append(generator_loss[2])
                discriminative_acc.append(d_loss[1] * 100)
                pbar.set_description(f"Epoch {epoch}/{epochs} - Batch {batch_i + 1}/{steps_per_epoch} - "
                                     f"[D_loss = {np.mean(discriminative_loss):.4f}"
                                     f" D_acc = {np.mean(discriminative_acc):.1f}%"
                                     f" E_loss = {np.mean(encoder_loss):.4f}"
                                     f" E_acc = {np.mean(encoder_acc):.1f}%"
                                     f" G_loss = {np.mean(decoder_loss):.4f}]")
                pbar.update(1)

            self.sample_images(dataset, val_generator, epoch, labels, denormalizer=denormalizer)
            with open(log_filename, 'aw'[epoch == 0]) as file:
                if epoch == 0:
                    file.write(f"Asc Time;Epoch;Discriminator loss;Discriminator accuracy;"
                               f"Encoder loss;Encoder class accuracy;Decoder loss\n")
                file.write(f"{time.asctime()};{epoch};{np.mean(discriminative_loss)};{np.mean(discriminative_acc)};"
                           f"{np.mean(encoder_loss)};{np.mean(encoder_acc)};{np.mean(decoder_loss)}\n")

            self.discriminator.save(self.DISCRIMINATOR_FILENAME)
            self.encoder.save(self.ENCODER_FILENAME)
            self.decoder.save(self.DECODER_FILENAME)

    def sample_images(self, dataset, generator, epoch, labels, denormalizer=None):
        os.makedirs(f"output/AEGAN_{dataset}", exist_ok=True)
        batch_input, batch_output = next(generator)
        latent_batch = self.encoder.predict(batch_input)
        prediction = [latent_batch[-1], self.decoder.predict(latent_batch)]

        callbacks.plot_example(batch_input, prediction, labels, batch_output, denormalizer=denormalizer)
        plt.savefig(os.path.join(ROOT, f"{epoch}.png"))
        plt.close()


def build_discriminator(img_shape):
    if img_shape[-1] == 1:
        img_input = Input(shape=img_shape)
        img_conc = concatenate([img_input, img_input, img_input])
        base_model = InceptionV3(input_tensor=img_conc, classes=1, include_top=False)
    else:
        base_model = InceptionV3(input_shape=img_shape, classes=1, include_top=False)
        img_input = base_model.input

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1, activation='sigmoid')(x)
    model = Model(inputs=img_input.input, outputs=x, name="discriminator")
    return model


if __name__ == '__main__':
    dataset = sys.argv[1]
    ROOT = f"output/AEGAN_{dataset}"
    gan = AEGAN((128, 128, 3), n_classes=10)  # TODO: n_classes computed
    gan.train(dataset, epochs=50, batch_size=4, steps_per_epoch=10)
