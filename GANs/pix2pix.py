import os, sys
import datetime
import matplotlib.pyplot as plt
import sys
from data_loader import DataLoader
import numpy as np

if __name__ == '__main__':
    sys.path.append(os.path.join(os.path.dirname(__file__), "..", '..'))
    os.environ['CUDA_VISIBLE_DEVICES'] = '2'

from keras.layers import Input, Dropout, Concatenate
from keras.layers import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Model, load_model
from keras.optimizers import Adam


class Pix2Pix():
    def __init__(self, dataset):
        # Input shape
        self.img_rows = 256
        self.img_cols = 256
        self.channels = 3
        self.img_shape = (self.img_rows, self.img_cols, self.channels)

        # Configure data loader
        self.dataset_name = dataset
        self.data_loader = DataLoader(dataset_name=self.dataset_name,
                                      img_res=(self.img_rows, self.img_cols))

        # Calculate output shape of D (PatchGAN)
        patch = int(self.img_rows / 2 ** 4)
        self.disc_patch = (patch, patch, 1)

        # Number of filters in the first layer of G and D
        self.gf = 64
        self.df = 64

        optimizer = Adam(0.0002, 0.5)

        self.DISCRIMINATOR_FILENAME = os.path.join(ROOT, 'discriminator.h5')
        if os.path.exists(self.DISCRIMINATOR_FILENAME):
            self.discriminator = load_model(self.DISCRIMINATOR_FILENAME)
        else:
            self.discriminator = build_discriminator(img_shape=self.img_shape, df=self.df)
            self.discriminator.compile(loss='mse', optimizer=optimizer, metrics=['accuracy'])

        self.GENERATOR_FILENAME = os.path.join(ROOT, 'decoder.h5')
        if os.path.exists(self.GENERATOR_FILENAME):
            self.generator = load_model(self.GENERATOR_FILENAME)
        else:
            self.generator = build_generator(img_shape=self.img_shape, gf=self.gf, channels=self.channels)

        # Input images and their conditioning images
        img_A = Input(shape=self.img_shape)
        img_B = Input(shape=self.img_shape)

        # By conditioning on B generate a fake version of A
        fake_A = self.generator(img_B)

        # For the combined model we will only train the generator
        self.discriminator.trainable = False

        # Discriminators determines validity of translated images / condition pairs
        valid = self.discriminator([fake_A, img_B])

        self.combined = Model(inputs=[img_A, img_B], outputs=[valid, fake_A], name="combined")
        self.combined.compile(loss=['mse', 'mae'], loss_weights=[1, 10], optimizer=optimizer)

    def train(self, epochs=100, batch_size=1, sample_interval=50):
        def process_batch(imgs_A, imgs_B):
            # Condition on B and generate a translated version
            fake_A = self.generator.predict(imgs_B)

            # Train the discriminators (original images = real / generated = Fake)
            d_loss_real = self.discriminator.train_on_batch([imgs_A, imgs_B], valid)
            d_loss_fake = self.discriminator.train_on_batch([fake_A, imgs_B], fake)
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

            g_loss = self.combined.train_on_batch([imgs_A, imgs_B], [valid, imgs_A])
            return d_loss, g_loss

        start_time = datetime.datetime.now()

        # Adversarial loss ground truths
        valid = np.ones((batch_size,) + self.disc_patch)
        fake = np.zeros((batch_size,) + self.disc_patch)

        for epoch in range(epochs):
            for batch_i, (imgs_A, imgs_B) in enumerate(self.data_loader.load_batch(batch_size)):
                d_loss, g_loss = process_batch(imgs_A, imgs_B)
                elapsed_time = datetime.datetime.now() - start_time
                # Plot the progress
                print(f"[Epoch {epoch}/{epochs}] [Batch {batch_i}/{self.data_loader.n_batches}] "
                      f"[D loss: {d_loss[0]:.3f}, acc: {100 * d_loss[1]:.1f}%] [G loss: {g_loss[0]:.3f}] "
                      f"time: {elapsed_time}")

                # If at save interval => save generated image samples
                if batch_i % sample_interval == 0:
                    self.sample_images(epoch, batch_i, batch_size)

            self.discriminator.save(self.DISCRIMINATOR_FILENAME)
            self.generator.save(self.GENERATOR_FILENAME)

    def sample_images(self, epoch, batch_i, batch_size):
        os.makedirs(f"output/pix2pix_{self.dataset_name}", exist_ok=True)
        r, c = 3, 6

        index = np.random.randint(0, self.data_loader.n_batches * batch_size, size=c)
        imgs_A = self.data_loader.load_data(domain="A", batch_size=r, is_testing=True, index=index)
        imgs_B = self.data_loader.load_data(domain="B", batch_size=r, is_testing=True, index=index)
        fake_A = self.generator.predict(imgs_B)

        gen_imgs = np.concatenate([imgs_B, fake_A, imgs_A])

        # Rescale images 0 - 1
        gen_imgs = 0.5 * gen_imgs + 0.5

        titles = ['Condition', 'Generated', 'Original']
        fig, axs = plt.subplots(r, c, figsize=(20, 10))
        cnt = 0
        for i in range(r):
            for j in range(c):
                axs[i, j].imshow(gen_imgs[cnt])
                axs[i, j].set_title(titles[i])
                axs[i, j].axis('off')
                cnt += 1
        fig.savefig(f"output/pix2pix_{self.dataset_name}/{epoch}_{batch_i}.png")
        plt.close()


def build_generator(img_shape, gf, channels):
    """U-Net Generator"""

    def conv2d(layer_input, filters, f_size=4, bn=True):
        """Layers used during downsampling"""
        d = Conv2D(filters, kernel_size=f_size, strides=2, padding='same')(layer_input)
        d = LeakyReLU(alpha=0.2)(d)
        if bn:
            d = BatchNormalization(momentum=0.8)(d)
        return d

    def deconv2d(layer_input, skip_input, filters, f_size=4, dropout_rate=0):
        """Layers used during upsampling"""
        u = UpSampling2D(size=2)(layer_input)
        u = Conv2D(filters, kernel_size=f_size, strides=1, padding='same', activation='relu')(u)
        if dropout_rate:
            u = Dropout(dropout_rate)(u)
        u = BatchNormalization(momentum=0.8)(u)
        u = Concatenate()([u, skip_input])
        return u

    # Image input
    d0 = Input(shape=img_shape)

    # Downsampling
    d1 = conv2d(d0, gf, bn=False)
    d2 = conv2d(d1, gf * 2)
    d3 = conv2d(d2, gf * 4)
    d4 = conv2d(d3, gf * 8)
    d5 = conv2d(d4, gf * 8)
    d6 = conv2d(d5, gf * 8)
    d7 = conv2d(d6, gf * 8)

    # Upsampling
    u1 = deconv2d(d7, d6, gf * 8)
    u2 = deconv2d(u1, d5, gf * 8)
    u3 = deconv2d(u2, d4, gf * 8)
    u4 = deconv2d(u3, d3, gf * 4)
    u5 = deconv2d(u4, d2, gf * 2)
    u6 = deconv2d(u5, d1, gf)

    u7 = UpSampling2D(size=2)(u6)
    output_img = Conv2D(channels, kernel_size=4, strides=1, padding='same', activation='tanh')(u7)

    return Model(d0, output_img, name="generator")


def build_discriminator(img_shape, df, inputs=2):
    def d_layer(layer_input, filters, f_size=4, bn=True):
        """Discriminator layer"""
        d = Conv2D(filters, kernel_size=f_size, strides=2, padding='same')(layer_input)
        d = LeakyReLU(alpha=0.2)(d)
        if bn:
            d = BatchNormalization(momentum=0.8)(d)
        return d

    if inputs == 2:
        img_A = Input(shape=img_shape)
        img_B = Input(shape=img_shape)
        inputs = [img_A, img_B]

        # Concatenate image and conditioning image by channels to produce input
        combined_imgs = Concatenate(axis=-1)([img_A, img_B])
    else:
        combined_imgs = Input(shape=img_shape)
        inputs = [combined_imgs]

    d1 = d_layer(combined_imgs, df, bn=False)
    d2 = d_layer(d1, df * 2)
    d3 = d_layer(d2, df * 4)
    d4 = d_layer(d3, df * 8)

    validity = Conv2D(1, kernel_size=4, strides=1, padding='same')(d4)

    return Model(inputs, validity, name="discriminator")


if __name__ == '__main__':
    dataset = sys.argv[1]
    ROOT = f"output/pix2pix_{dataset}"
    gan = Pix2Pix(dataset)
    gan.train(epochs=200, batch_size=16, sample_interval=200)
