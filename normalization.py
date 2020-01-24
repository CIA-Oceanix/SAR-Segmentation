import numpy as np
from Rignak_ImageProcessing.miscellaneous_image_operations import fourier_transform, inverse_fourier_transform


def intensity_normalization(f=1 / 255):
    def denormalizer(batch):
        batch /= f
        if batch.max() > 1:
            batch = batch.astype('uint8')
        return batch

    return lambda x: x * f, denormalizer


def tanh_normalization(f=255):
    return lambda batch: (batch / f * 2) - 1, lambda batch: ((batch + 1) / 2) * f


def log_normalization(delta=0.001):
    return lambda batch: np.log(batch + delta), lambda batch: np.exp(batch)


def fake_normalization():
    return lambda batch: batch, lambda batch: batch


def fourier_normalization():
    def image_normalization(im):
        if len(im.shape) == 2:
            im = np.expand_dims(im, axis=-1)
        new_im = np.zeros((im.shape[0], im.shape[1], im.shape[2] * 3))
        for canal in range(im.shape[2]):
            new_im[:, :, 3 * canal] = im[:, :, canal]
            ft, new_im[:, :, 3 * canal + 1], new_im[:, :, 3 * canal + 2] = fourier_transform(im[:, :, canal],
                                                                                             remove_center=True)
        return new_im

    def normalize(batch):
        if len(batch.shape) == 4:
            batch = np.mean(batch, axis=-1)
            batch = np.expand_dims(batch, axis=-1)

            new_batch = np.zeros((batch.shape[0], batch.shape[1], batch.shape[2], batch.shape[3] * 3))
            for i, frequency_image in enumerate(batch):
                new_batch[i] = image_normalization(frequency_image)
        else:
            new_batch = image_normalization(batch)
        return new_batch

    def image_denormalization(im):
        dim = im.shape[1]
        concatenation = np.zeros((im.shape[0], dim * 2, im.shape[-1] // 3))

        for canal in range(concatenation.shape[-1]):
            frequency_canal = im[:, :, canal + 1] + im[:, :, canal + 2] * 1j
            concatenation[:, :dim, canal] = inverse_fourier_transform(frequency_canal, m=im[:, :, canal])
            concatenation[:, dim:, canal] = np.abs(frequency_canal)

        concatenation[:, dim:] /= concatenation[:, dim:].max()
        concatenation[:, :dim] /= concatenation[:, :dim].max()
        return concatenation

    def denormalize(batch):
        if len(batch.shape) == 4:
            new_batch = np.zeros((batch.shape[0], batch.shape[1], batch.shape[2], batch.shape[3] // 3))
            for i, frequency_image in enumerate(batch):
                new_batch[i] = image_denormalization(frequency_image)
        else:
            new_batch = image_denormalization(batch)
        return new_batch

    return normalize, denormalize


NORMALIZATION_FUNCTIONS = {'intensity': intensity_normalization,
                           'tanh': tanh_normalization,
                           'log': log_normalization,
                           'none': fake_normalization,
                           'fourier': fourier_normalization}
