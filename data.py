import os
import numpy as np
from PIL import Image
import cv2
import scipy.misc

from keras.preprocessing import image as image_utils

CANALS = 3


def thumbnail(filename, size):
    with Image.open(filename) as image:
        image.thumbnail(size)
        width, height = image.size
        new_image = Image.new('RGB', size, (0, 0, 0))
        new_image.paste(image, ((new_image.size[0] - width) // 2,
                                (new_image.size[1] - height) // 2))
    return image_utils.img_to_array(new_image)


def data_on_folder(folder, size, canals=CANALS):
    filenames = os.listdir(folder)
    filenames = [os.path.join(folder, filename) for filename in filenames]
    array = np.zeros((len(filenames), size[0], size[1], canals))
    for i, filename in enumerate(filenames):
        array[i] = thumbnail(filename, size)
    return array, filenames


def read(filename, input_shape):
    im = cv2.imread(filename)
    if im is None:
        print(f"Error when reading {filename}")
        return np.zeros(input_shape)
    if im.shape != input_shape:
        im = scipy.misc.imresize(im, input_shape[:2])
    return im


def get_dataset_roots(task, dataset='.'):
    if task in ['categorizer', 'saliency']:
        train_root = os.path.join('E:', 'datasets', 'categorizer', dataset, 'train')
        val_root = os.path.join('E:', 'datasets', 'categorizer', dataset, 'val')
    elif task in ['autoencoder', 'style_transfer']:
        train_root = os.path.join('E:', 'datasets', task, dataset, 'train')
        val_root =os.path.join('E:', 'datasets', task, dataset, 'val')
    elif task in ['stylegan', 'cyclegan']:
        train_root = os.path.join('E:', 'datasets', task, dataset)
        val_root = os.path.join('E:', 'datasets', task, dataset)
    else:
        raise ValueError
    return train_root, val_root
