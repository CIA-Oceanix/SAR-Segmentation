import os, sys
import sys
import matplotlib.pyplot as plt
import numpy as np
import tqdm

if __name__ == '__main__':
    sys.path.append(os.path.join(os.path.dirname(__file__), "..", '..'))
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

from keras.applications.inception_v3 import InceptionV3
from keras.models import Model, load_model
from keras.layers import Conv2D, Lambda
from keras.layers import Input, Dense, GlobalAveragePooling2D, concatenate, Reshape, UpSampling2D, Conv2DTranspose
from keras.optimizers import Adam

from Rignak_DeepLearning.models import convolution_block
from Rignak_DeepLearning.BiOutput import generator, callbacks
from Rignak_DeepLearning.data import get_dataset_roots

from Rignak_DeepLearning.GANs.pix2pix import build_discriminator as build_pixiv_discriminator


class AEGAN():
    def __init__(self, image_shape, n_classes):
        self.img_rows = image_shape[0]
        self.img_cols = image_shape[1]
        self.channels = image_shape[2]
        self.img_shape = image_shape

        self.latent_space_root_length = 8
        self.n_classes = n_classes
        self.convs = [32, 64, 128, 128, 128, 256]
        optimizer = Adam(0.0002, 0.5)

        patch = int(self.img_rows / 2 ** 4)
        self.disc_patch = (patch, patch, 1)

        self.DISCRIMINATOR_FILENAME = os.path.join(ROOT, 'discriminator.h5')
        self.ENCODER_FILENAME = os.path.join(ROOT, 'encoder.h5')
        self.DECODER_FILENAME = os.path.join(ROOT, 'decoder.h5')

        if os.path.exists(self.DISCRIMINATOR_FILENAME):
            self.discriminator = load_model(self.DISCRIMINATOR_FILENAME)
        else:
            # self.discriminator = build_discriminator(img_shape=self.img_shape)
            # self.discriminator.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
            self.discriminator = build_pixiv_discriminator(img_shape=self.img_shape, df=64, inputs=1)
            self.discriminator.compile(loss='mse', optimizer=optimizer, metrics=['accuracy'])

        if os.path.exists(self.ENCODER_FILENAME):
            self.encoder = load_model(self.ENCODER_FILENAME)
        else:
            self.encoder = build_encoder(img_shape=self.img_shape, convs=self.convs, n_classes=self.n_classes,
                                         latent_space_root_length=self.latent_space_root_length)

        if os.path.exists(self.DECODER_FILENAME):
            self.decoder = load_model(self.DECODER_FILENAME)
        else:
            self.decoder = build_decoder(n_classes=self.n_classes, convs=self.convs, img_shape=self.img_shape,
                                         latent_space_root_length=self.latent_space_root_length)

        self.classifier = build_classifier(n_classes=self.n_classes, convs=self.convs,
                                           latent_space_root_length=self.latent_space_root_length)
        self.classifier.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['acc'])

        img = Input(shape=self.img_shape)
        encoded_repr = self.encoder(img)
        encoded_class = self.classifier(encoded_repr)
        reconstructed_img = self.decoder(encoded_repr)

        self.discriminator.trainable = False
        self.classifier.trainable = False
        validity = self.discriminator(reconstructed_img)

        self.adversarial_autoencoder = Model(img, [encoded_class, reconstructed_img, validity], name="adversarial")
        self.adversarial_autoencoder.compile(loss=['categorical_crossentropy', 'mae', 'mse'],
                                             loss_weights=[1, 1, 0.05], optimizer=optimizer)

    def train(self, dataset, epochs, batch_size=16, sample_interval=300, steps_per_epoch=200):
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
            generator_loss = self.adversarial_autoencoder.train_on_batch(batch_input,
                                                                         [batch_label, batch_output, valid])
            return generator_loss, correct_classification_rate, d_loss

        # Adversarial loss ground truths
        valid = np.ones((batch_size,) + self.disc_patch)
        fake = np.zeros((batch_size,) + self.disc_patch)

        train_folder, val_folder = get_dataset_roots('', dataset=dataset)
        labels = [folder for folder in os.listdir(train_folder)
                  if os.path.isdir(os.path.join(train_folder, folder))]
        train_generator = generator.generator(train_folder, batch_size=batch_size, input_shape=self.img_shape)
        val_generator = generator.generator(val_folder, batch_size=batch_size, input_shape=self.img_shape)

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

                if batch_i % sample_interval == 0:
                    self.sample_images(dataset, val_generator, epoch, batch_i, labels)

            self.discriminator.save(self.DISCRIMINATOR_FILENAME)
            self.encoder.save(self.ENCODER_FILENAME)
            self.decoder.save(self.DECODER_FILENAME)

    def sample_images(self, dataset, generator, epoch, batch_i, labels):
        os.makedirs(f"output/AEGAN_{dataset}", exist_ok=True)
        batch_input, batch_output = next(generator)
        latent_batch = self.encoder.predict(batch_input)
        prediction = [self.classifier.predict(latent_batch), self.decoder.predict(latent_batch)]

        callbacks.plot_example(batch_input, prediction, labels, batch_output)
        plt.savefig(os.path.join(ROOT, f"{epoch}_{batch_i}.png"))
        plt.close()


def build_classifier(n_classes, latent_space_root_length, convs):
    inputs = [Input([n_classes])] + [Input([latent_space_root_length ** 2]) for _ in convs[:-1]] + \
             [Input([latent_space_root_length ** 2 - n_classes])]
    class_layer = Lambda(lambda x: x, name='endoded_class')(inputs[0])
    model = Model(inputs=inputs, outputs=class_layer, name="classifier")
    return model


def build_discriminator(img_shape):
    if img_shape[-1] == 1:
        img_input = Input(shape=img_shape)
        img_conc = Concatenate()([img_input, img_input, img_input])
        base_model = InceptionV3(input_shape=(img_shape[0], img_shape[2], 3), classes=1, include_top=False)
    else:
        base_model = InceptionV3(input_shape=img_shape, classes=1, include_top=False)
        img_input = base_model.input

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1, activation='sigmoid')(x)
    model = Model(inputs=img_input.input, outputs=x, name="discriminator")
    return model


def build_encoder(img_shape, convs, n_classes, latent_space_root_length):
    input_layer = Input(img_shape, name="input")
    latent_space_layers = []
    block = None
    for i, neurons in enumerate(convs):
        if block is None:
            block, _ = convolution_block(input_layer, neurons, activation='selu', batch_normalization=True)
        else:
            block, _ = convolution_block(block, neurons, activation='selu', batch_normalization=True)

        global_pooling = GlobalAveragePooling2D()(block)
        latent_space_layer = Dense(neurons, activation='selu')(global_pooling)
        if i == len(convs) - 1:
            neurons = latent_space_root_length ** 2 - n_classes
        else:
            neurons = latent_space_root_length ** 2
        latent_space_layer = Dense(neurons, activation='selu')(latent_space_layer)
        latent_space_layers.append(latent_space_layer)

    categorization_layer = Dense(n_classes, activation='softmax')(global_pooling)
    latent_space_layers = [categorization_layer] + latent_space_layers
    model = Model(inputs=input_layer, outputs=latent_space_layers, name="encoder")
    return model


def build_decoder(n_classes, latent_space_root_length, convs, img_shape):
    inputs = [Input([n_classes])] + [Input([latent_space_root_length ** 2]) for _ in convs[:-1]] + \
             [Input([latent_space_root_length ** 2 - n_classes])]
    latent_space_layers = [layer for layer in inputs[1:-1]]
    latent_space_layers.append(concatenate([inputs[0], inputs[-1]]))

    latent_space_layers = [Reshape((latent_space_root_length, latent_space_root_length, 1),
                                   name=f"decoder_input{i}")(layer) for i, layer in enumerate(latent_space_layers)]

    block = None
    for i, (neurons, latent_space_layer) in enumerate(zip(convs[::-1], latent_space_layers[::-1])):
        if block is not None:
            upsample = UpSampling2D((2 ** i, 2 ** i))(latent_space_layer)
            block = concatenate([upsample, block])
        else:
            block = latent_space_layer
        block, _ = convolution_block(block, neurons, activation='selu', maxpool=False, batch_normalization=True)
        if i != len(convs) - 1:
            # block = UpSampling2D((2, 2))(block)
            block = Conv2DTranspose(neurons, strides=(2, 2), kernel_size=(3, 3),
                                    padding='same', activation='selu')(block)
    decoder_layer = Conv2D(img_shape[-1], (1, 1), activation='sigmoid')(block)
    model = Model(inputs=inputs, outputs=decoder_layer, name="decoder")
    return model


if __name__ == '__main__':
    dataset = sys.argv[1]
    ROOT = f"output/AEGAN_{dataset}"
    gan = AEGAN((256, 256, 1), n_classes=10)
    gan.train(dataset, epochs=10000, batch_size=8)
