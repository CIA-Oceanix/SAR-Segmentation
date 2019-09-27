import numpy as np


def get_uniform_noise_function(f=0.35):
    def uniform_noise(x):  # std is around 0.35
        xmax = np.max(x)
        xmin = np.min(x)
        noise = f * x.std() * (2 * np.random.random(x.shape) - 1)
        x = x.astype('float64')
        x += noise
        x[x > xmax] = xmax
        x[x < xmin] = xmin
        return x

    return uniform_noise


def get_disable_pixel_function(f=0.33):
    def disable_pixel(x):
        for i in range(x.shape[0]):
            r = np.random.random(x.shape[2])
            x[i, r < f] = 0
            r = np.random.random(x.shape[2])
            x[i, :, r < f] = 0
        return x

    return disable_pixel
