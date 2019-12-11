import numpy as np

DEFAULT_NOSE = 0.0
DEFAULT_DISABLE_PIXEL = 1/3

def get_uniform_noise_function(f=DEFAULT_NOSE):
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


def get_disable_pixel_function(f=DEFAULT_DISABLE_PIXEL):
    def disable_pixel(x):
        for i in range(x.shape[0]):
            r = np.random.random(x.shape[2])
            x[i, r < f] = 0
            r = np.random.random(x.shape[2])
            x[i, :, r < f] = 0
        return x

    return disable_pixel
