import os, sys
import matplotlib.pyplot as plt
import numpy as np
import tqdm
import time

if __name__ == '__main__':
    sys.path.append(os.path.join(os.path.dirname(__file__), "..", '..'))
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

from keras.applications.inception_v3 import InceptionV3
from keras.models import Model, load_model
from keras.layers import Conv2D, Lambda, Flatten
from keras.layers import Input, Dense, GlobalAveragePooling2D, concatenate, Reshape, Conv2DTranspose
from keras.optimizers import Adam
from keras_radam.training import RAdamOptimizer

from Rignak_Misc.path import get_local_file
from Rignak_DeepLearning.normalization import tanh_normalization, log_normalization, fake_normalization, intensity_normalization
from Rignak_DeepLearning.models import convolution_block
from Rignak_DeepLearning.BiOutput import generator, callbacks
from Rignak_DeepLearning.data import get_dataset_roots
from Rignak_DeepLearning.GANs.pix2pix import build_discriminator as build_pix2pix_discriminator

LOG_FILENAME = get_local_file(__file__, "AEGAN_LOGS.csv")


class AEGAN():
    def __init__(self, image_shape, dataset):
        self.img_rows, self.img_cols, self.channels = image_shape
        self.img_shape = image_shape

        self.latent_space_shape = (4, 4, 64)

        train_folder, val_folder = get_dataset_roots('', dataset=dataset)
        n_classes = len([folder for folder in os.listdir(train_folder)
                         if os.path.isdir(os.path.join(train_folder, folder))])

        self.n_classes = n_classes
        print('n_classes', n_classes)
        self.convs = [32, 64, 64, 128]
        optimizer = RAdamOptimizer(1e-3)

        patch = int(self.img_rows / 2 ** 4)
        self.disc_patch = (patch, patch, 1)

        self.DISCRIMINATOR_FILENAME = os.path.join(ROOT, 'discriminator.h5')
        self.ENCODER_FILENAME = os.path.join(ROOT, 'encoder.h5')
        self.DECODER_FILENAME = os.path.join(ROOT, 'decoder.h5')
        self.ADVERSARIAL_FILENAME = os.path.join(ROOT, 'adversarial.h5')

        self.classifier = build_classifier(n_classes=self.n_classes, latent_space_shape=self.latent_space_shape)
        self.classifier.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['acc'])

        if os.path.exists(self.DISCRIMINATOR_FILENAME):
            self.discriminator = load_model(self.DISCRIMINATOR_FILENAME)
        else:
            # self.discriminator = build_discriminator(img_shape=self.img_shape)
            # self.discriminator.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
            self.discriminator = build_pix2pix_discriminator(img_shape=self.img_shape, df=64, inputs=1)
        self.discriminator.compile(loss='mse', optimizer=optimizer, metrics=['accuracy'])

        if os.path.exists(self.ENCODER_FILENAME):
            self.encoder = load_model(self.ENCODER_FILENAME)
        else:
            self.encoder = build_encoder(img_shape=self.img_shape, convs=self.convs, n_classes=self.n_classes,
                                         latent_space_shape=self.latent_space_shape)

        if os.path.exists(self.DECODER_FILENAME):
            self.decoder = load_model(self.DECODER_FILENAME)
        else:
            self.decoder = build_decoder(img_shape=self.img_shape, n_classes=self.n_classes,
                                         latent_space_shape=self.latent_space_shape, convs=self.convs)

        img = Input(shape=self.img_shape)
        encoded_repr = self.encoder(img)
        encoded_class = self.classifier(encoded_repr)
        reconstructed_img = self.decoder(encoded_repr)

        self.discriminator.trainable = False
        self.classifier.trainable = False
        validity = self.discriminator(reconstructed_img)
        self.adversarial_autoencoder = Model(img, [encoded_class, reconstructed_img, validity], name="adversarial")
        self.adversarial_autoencoder.compile(loss={'classifier': 'categorical_crossentropy',
                                                   'decoder': 'mse',
                                                   'discriminator': 'mse'},
                                             loss_weights={"classifier": 1.0, "decoder": 0.0, "discriminator": 0.0},
                                             optimizer='adam')

    def train(self, dataset, epochs, batch_size=8, steps_per_epoch=200,
              log_filename=LOG_FILENAME):
        def process_batch(generator):
            batch_input, (batch_label, batch_output) = next(generator)

            latent_batch = self.encoder.predict(batch_input)
            reconstructed_batch = self.decoder.predict(latent_batch)

            predicted_labels = latent_batch[0]
            correct_classification_rate = np.mean([int(np.argmax(pred) == np.argmax(truth))
                                                   for pred, truth in zip(batch_label, predicted_labels)])

            # Train the discriminator
            d_loss = self.discriminator.train_on_batch(batch_input, valid)
            d_loss += self.discriminator.train_on_batch(reconstructed_batch, fake)

            # Train the generator
            generator_loss = self.adversarial_autoencoder.train_on_batch(batch_input, [batch_label, batch_output, valid])
            return generator_loss, correct_classification_rate, d_loss

        # Adversarial loss ground truths
        valid = np.ones((batch_size,) + self.disc_patch)
        fake = np.zeros((batch_size,) + self.disc_patch)

        # Get generators
        normalizer, denormalizer = intensity_normalization(f=1/255)
        train_folder, val_folder = get_dataset_roots('', dataset=dataset)
        labels = [folder for folder in os.listdir(train_folder)
                  if os.path.isdir(os.path.join(train_folder, folder))]
        train_generator = generator.generator(train_folder, batch_size=batch_size, input_shape=self.img_shape)
        train_generator = generator.normalize_generator(train_generator, normalizer, apply_on_output=True)

        val_generator = generator.generator(val_folder, batch_size=4, input_shape=self.img_shape)
        val_generator = generator.normalize_generator(val_generator, normalizer, apply_on_output=True)

        self.adversarial_autoencoder.summary()

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
                pbar.set_description(f"Epoch {epoch}/{epochs} - Batch {batch_i}/{steps_per_epoch} - "
                                     f"[D_loss = {np.mean(discriminative_loss):.4f}"
                                     f" D_acc = {np.mean(discriminative_acc):.1f}%"
                                     f" E_loss = {np.mean(encoder_loss):.4f}"
                                     f" E_acc = {np.mean(encoder_acc):.1f}%"
                                     f" G_loss = {np.mean(decoder_loss):.4f}]")
                pbar.update(1)

            self.sample_images(dataset, val_generator, epoch, labels)
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
        prediction = [self.classifier.predict(latent_batch), self.decoder.predict(latent_batch)]

        callbacks.plot_example(batch_input, prediction, labels, batch_output, denormalization=denormalizer)
        plt.savefig(os.path.join(ROOT, f"{epoch}.png"))
        plt.close()


def build_classifier(n_classes, latent_space_shape):
    inputs = [Input([n_classes]), Input([np.prod(latent_space_shape) - n_classes])]
    class_layer = Lambda(lambda x: x, name='endoded_class')(inputs[0])
    model = Model(inputs=inputs, outputs=class_layer, name="classifier")
    return model


def build_discriminator(img_shape):
    base_model = InceptionV3(input_shape=img_shape, classes=1, include_top=False)
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1, activation='sigmoid')(x)
    model = Model(inputs=base_model.input, outputs=x, name="discriminator")
    return model


def build_encoder(img_shape, convs, n_classes, latent_space_shape):
    input_layer = Input(img_shape, name="input")

    block = None
    for i, neurons in enumerate(convs):
        if block is None:
            block, _ = convolution_block(input_layer, neurons, activation='selu', batch_normalization=True)
        else:
            block, _ = convolution_block(block, neurons, activation='selu', batch_normalization=True)

    #block = GlobalAveragePooling2D()(block)
    block = Flatten()(block)
    latent_space_layer = Dense(neurons, activation='selu')(block)
    categorization_layer = Dense(n_classes, activation='softmax')(latent_space_layer)
    latent_space_layer = Dense(np.prod(latent_space_shape) - n_classes, activation='selu')(latent_space_layer)

    model = Model(inputs=input_layer, outputs=[categorization_layer, latent_space_layer], name="encoder")
    print('ENCODER MODEL')
    model.summary()
    return model


def build_decoder(img_shape, n_classes, latent_space_shape, convs):
    inputs = [Input([n_classes]), Input([np.prod(latent_space_shape) - n_classes])]

    block = concatenate([inputs[0], inputs[-1]])
    block = Reshape(latent_space_shape)(block)
    for i, neurons in enumerate(convs[::-1]):
        block = Conv2DTranspose(neurons, strides=(2, 2), kernel_size=(4, 4), padding='same', activation='selu')(block)
        block, _ = convolution_block(block, neurons, activation='selu', maxpool=False, batch_normalization=True)

    decoder_layer = Conv2D(img_shape[-1], (1, 1), activation='sigmoid')(block)
    model = Model(inputs=inputs, outputs=decoder_layer, name="decoder")
    print('DECODER MODEL')
    model.summary()
    return model


if __name__ == '__main__':
    dataset = sys.argv[1]
    ROOT = f"output/AEGAN_V0_{dataset}"
    gan = AEGAN((128, 128, 3), dataset=dataset)
    gan.train(dataset, epochs=10000, batch_size=1)
