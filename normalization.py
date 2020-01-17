import numpy as np


def intensity_normalization(f=1 / 1):
    def normalize(im):
        return im * f

    def denormalize(im):
        return im / f

    return normalize, denormalize


def tanh_normalization(f=255):
    def normalize(im):
        return (im / f * 2) - 1

    def denormalize(im):
        return ((im + 1) / 2) * f

    return normalize, denormalize


def log_normalization(delta=0.001):
    def normalize(im):
        im[im == 0] += delta
        return np.log(im)

    def denormalize(im):
        return np.exp(im)

    return normalize, denormalize

def fake_normalization():
    def normalize(im):
        return im

    def denormalize(im):
        return im

    return normalize, denormalize