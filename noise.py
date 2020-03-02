import numpy as np
import scipy

DEFAULT_NOISE = 1.0
DEFAULT_CATEGORISATION_NOISE = 0.2
DEFAULT_DISABLE_PIXEL = 1 / 3
DEFAULT_CONTRAST = 0.5

DEFAULT_DECREASING_SIZE = 12

DEFAULT_FIRST_DECREASING_PERIOD = 20000
DEFAULT_DECREASING_PERIOD = 2000


def get_uniform_noise_function(f=DEFAULT_NOISE):
    def uniform_noise(x, y):  # std is around 0.35
        xmax = np.max(x)
        xmin = np.min(x)
        noise = f * x.std() * (2 * np.random.random(x.shape) - 1)
        x = x.astype('float64')
        x += noise
        x = np.maximum(x, xmin)
        x = np.minimum(x, xmax)
        return x, y

    return uniform_noise


def get_uniform_output_noise_function(f=DEFAULT_NOISE):
    def uniform_noise(x, y):  # std is around 0.35
        ymax = np.max(y)
        ymin = np.min(y)
        noise = f * y.std() * (2 * np.random.random(y.shape) - 1)
        y = y.astype('float64')
        y += noise
        y = np.maximum(y, ymin)
        y = np.minimum(y, ymax)
        return x, y

    return uniform_noise


def get_disable_pixel_function(f=DEFAULT_DISABLE_PIXEL):
    def disable_pixel(x, y):
        for i in range(x.shape[0]):
            r = np.random.random(x.shape[2])
            x[i, r < f] = 0
            r = np.random.random(x.shape[2])
            x[i, :, r < f] = 0
        return x, y

    return disable_pixel


def get_contrast_noise_function(f=DEFAULT_CONTRAST):
    def contrast_noise_function(x, y):
        xmax = np.max(x)
        xmin = np.min(x)
        random_f = 1 + f * (np.random.random() - 0.5) * 2
        new_x = ((x / xmax) ** random_f) * xmax
        new_x = np.maximum(new_x, xmin)
        new_x = np.minimum(new_x, xmax)
        return new_x, y

    return contrast_noise_function


def get_categorization_noise_function(f=DEFAULT_CATEGORISATION_NOISE):
    def categorization_noise_function(x, y):
        random_f = f * (np.random.random(y.shape) - 0.5) * 2
        new_y = y + random_f
        new_y = np.maximum(new_y, 0)
        new_y = np.minimum(new_y, 1)
        return x, new_y

    return categorization_noise_function


def get_decreasing_contacts(first_period=DEFAULT_FIRST_DECREASING_PERIOD, period=DEFAULT_DECREASING_PERIOD,
                            kernel_size=DEFAULT_DECREASING_SIZE):
    def get_kernel():
        size = function_variables['kernel_radius']
        x, y = np.meshgrid(np.linspace(-size, size, 2 * size), np.linspace(-size, size, 2 * size))
        d = np.sqrt(x * x + y * y)
        kernel = np.maximum(0, 1 - d / size)
        kernel = (kernel - kernel.min()) / kernel.max()
        function_variables['kernel'] = kernel

    def decreasing_contacts(x, y):
        if function_variables['kernel_radius'] <= 1:
            return x, y

        ymax = y.max()
        y = y / ymax
        structure = function_variables['kernel']
        for i in range(y.shape[0]):
            y[i] = scipy.ndimage.morphology.grey_dilation(y[i], structure=structure[:, :, None]) - 1
        y = (y - y.min()) * ymax

        function_variables['batch'] += 1
        if not function_variables['batch'] % function_variables['period']:
            function_variables['kernel_radius'] -= 1
            function_variables['period'] = period
            print('\nDecreasing kernel size to', function_variables['kernel_radius'])
            get_kernel()
        return x, y

    function_variables = {'batch': 0, 'kernel_radius': kernel_size, 'period': first_period}
    get_kernel()
    return decreasing_contacts


def get_composition(functions):
    def apply_composition(x, y):
        new_x = x
        new_y = y
        for function in functions:
            new_x, new_y = NOISE_FUNCTIONS[function](new_x, new_y)
        return new_x, new_y

    return apply_composition


def get_none_noise():
    return lambda x, y: (x, y)


NOISE_FUNCTIONS = {'uniform': get_uniform_noise_function(),
                   'contrast': get_contrast_noise_function(),
                   'composition': get_composition,
                   'categorization': get_categorization_noise_function(),
                   'decreasing_contacts': get_decreasing_contacts(),
                   'output_uniform': get_uniform_output_noise_function(),
                   None: get_none_noise()}
