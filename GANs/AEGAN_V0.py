import os
import sys
import matplotlib.pyplot as plt
import numpy as np
import tqdm

from keras.applications.inception_v3 import InceptionV3
from keras.models import Model, load_model
from keras.layers import Conv2D, Lambda
from keras.layers import Input, Dense, GlobalAveragePooling2D, concatenate, Reshape, Conv2DTranspose

from Rignak_DeepLearning.models import convolution_block
from Rignak_DeepLearning.BiOutput import generator, callbacks
from Rignak_DeepLearning.data import get_dataset_roots


class AEGAN():
    def __init__(self, image_shape, n_classes):
        self.img_rows = image_shape[0]
        self.img_cols = image_shape[1]
        self.channels = 1
        self.img_shape = image_shape

        self.latent_space_length = 256
        self.n_classes = n_classes
        self.convs = [128, 128, 128]

        self.DISCRIMINATOR_FILENAME = os.path.join(ROOT, 'discriminator.h5')
        self.ENCODER_FILENAME = os.path.join(ROOT, 'encoder.h5')
        self.DECODER_FILENAME = os.path.join(ROOT, 'decoder.h5')
        self.ADVERSARIAL_FILENAME = os.path.join(ROOT, 'adversarial.h5')

        self.classifier = self.build_classifier()
        self.classifier.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])

        if os.path.exists(self.DISCRIMINATOR_FILENAME):
            self.discriminator = load_model(self.DISCRIMINATOR_FILENAME)
        else:
            self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])

        if os.path.exists(self.ENCODER_FILENAME):
            self.encoder = load_model(self.ENCODER_FILENAME)
        else:
            self.encoder = self.build_encoder()

        if os.path.exists(self.DECODER_FILENAME):
            self.decoder = load_model(self.DECODER_FILENAME)
        else:
            self.decoder = self.build_decoder()

        img = Input(shape=self.img_shape)
        encoded_repr = self.encoder(img)
        encoded_class = self.classifier(encoded_repr)
        reconstructed_img = self.decoder(encoded_repr)

        self.discriminator.trainable = False
        self.classifier.trainable = False
        validity = self.discriminator(reconstructed_img)
        self.adversarial_autoencoder = Model(img, [encoded_class, reconstructed_img, validity], name="adversarial")
        self.adversarial_autoencoder.compile(loss=['categorical_crossentropy', 'mse', 'binary_crossentropy'],
                                             loss_weights=[1, 1, 0.0001],
                                             optimizer='adam')

    def build_classifier(self):
        inputs = [Input([self.n_classes]), Input([self.latent_space_length - self.n_classes])]
        class_layer = Lambda(lambda x: x, name='endoded_class')(inputs[0])
        model = Model(inputs=inputs, outputs=class_layer, name="classifier")
        return model

    def build_discriminator(self):
        base_model = InceptionV3(input_shape=self.img_shape, classes=1, include_top=False)
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        x = Dense(1, activation='sigmoid')(x)
        model = Model(inputs=base_model.input, outputs=x, name="discriminator")
        return model

    def build_encoder(self):
        input_layer = Input(self.img_shape, name="input")

        block = None
        for i, neurons in enumerate(self.convs):
            if block is None:
                block, _ = convolution_block(input_layer, neurons, activation='selu', batch_normalization=True)
            else:
                block, _ = convolution_block(block, neurons, activation='selu', batch_normalization=True)

        block = GlobalAveragePooling2D()(block)
        latent_space_layer = Dense(self.latent_space_length - self.n_classes, activation='selu')(block)
        categorization_layer = Dense(self.n_classes, activation='softmax')(block)

        model = Model(inputs=input_layer, outputs=[categorization_layer, latent_space_layer], name="encoder")
        return model

    def build_decoder(self):
        inputs = [Input([self.n_classes]), Input([self.latent_space_length - self.n_classes])]

        size = int(self.latent_space_length ** .5)
        block = concatenate([inputs[0], inputs[-1]])
        block = Reshape((size, size, 1))(block)
        for i, neurons in enumerate(self.convs[::-1]):
            block = Conv2DTranspose(neurons, strides=(2, 2), kernel_size=(4, 4), padding='same', activation='selu')(
                block)
            block, _ = convolution_block(block, neurons, activation='selu', maxpool=False, batch_normalization=True)

        decoder_layer = Conv2D(self.img_shape[-1], (1, 1), activation='sigmoid')(block)
        model = Model(inputs=inputs, outputs=decoder_layer, name="decoder")
        return model

    def train(self, dataset, epochs, batch_size=8, sample_interval=300, steps_per_epoch=200):
        # Adversarial loss ground truths
        valid = np.ones((batch_size, 1))
        fake = np.zeros((batch_size, 1))

        train_folder, val_folder = get_dataset_roots('', dataset=dataset)
        labels = [folder for folder in os.listdir(train_folder)
                  if os.path.isdir(os.path.join(train_folder, folder))]
        train_generator = generator.generator(train_folder, batch_size=batch_size, input_shape=self.img_shape)
        val_generator = generator.generator(val_folder, batch_size=batch_size, input_shape=self.img_shape)

        self.adversarial_autoencoder.summary()

        for epoch in range(epochs):
            pbar = tqdm.trange(steps_per_epoch)

            encoder_loss = []
            decoder_loss = []
            discriminative_loss = []
            discriminative_acc = []
            for batch_i in range(steps_per_epoch):
                batch_input, (batch_label, batch_output) = next(train_generator)
                batch_input = batch_input / 255
                batch_output = batch_output / 255

                latent_batch = self.encoder.predict(batch_input)
                reconstructed_batch = self.decoder.predict(latent_batch)

                # Train the discriminator
                d_loss = self.discriminator.train_on_batch(batch_input, valid)
                d_loss += self.discriminator.train_on_batch(reconstructed_batch, fake)

                # Train the generator
                generator_loss = self.adversarial_autoencoder.train_on_batch(batch_input,
                                                                             [batch_label, batch_output, valid])

                encoder_loss.append(generator_loss[0])
                decoder_loss.append(generator_loss[1])
                discriminative_loss.append(generator_loss[2])
                discriminative_acc.append(d_loss[1] * 100)
                pbar.set_description(f"Epoch {epoch}/{epochs} - Batch {batch_i}/{steps_per_epoch} - "
                                     f"[D_loss = {np.mean(discriminative_loss):.4f}"
                                     f" D_acc = {np.mean(discriminative_acc):.1f}%"
                                     f" E_loss = {np.mean(encoder_loss):.4f}"
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
        batch_input = batch_input / 255
        batch_output[1] = batch_output[1] / 255
        latent_batch = self.encoder.predict(batch_input)
        prediction = [self.classifier.predict(latent_batch), self.decoder.predict(latent_batch)]

        callbacks.plot_example(batch_input, prediction, labels, batch_output)
        plt.savefig(os.path.join(ROOT, f"/{epoch}_{batch_i}.png"))
        print(os.path.join(ROOT, f"/{epoch}_{batch_i}.png"))
        plt.close()


if __name__ == '__main__':
    dataset = sys.argv[1]
    ROOT = f"output/AEGAN_V0_{dataset}"
    gan = AEGAN((128, 128, 3), 8)
    gan.train(dataset, 100)
