import os
import numpy as np
from PIL import Image
import cv2
import scipy.misc
from tqdm import tqdm

import PIL.Image

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
    filenames = [os.path.join(folder, filename) for filename in os.listdir(folder)]
    array = np.zeros((len(filenames), size[0], size[1], canals))
    for i, filename in enumerate(filenames):
        array[i] = thumbnail(filename, size)
    return array, filenames


def read(filename, input_shape=None):
    with PIL.Image.open(filename) as im:
        im = np.array(im)

    if im is None:
        print(f"Error when reading {filename}")
        return np.zeros(input_shape)
    if input_shape is not None and im.shape != input_shape:
        im = scipy.misc.imresize(im, input_shape[:2])
    if len(im.shape) == 2:
        im = np.expand_dims(im, axis=-1)
    return im


def get_dataset_roots(task, dataset='.'):
    train_root = os.path.join('E:', 'datasets', dataset, 'train')
    val_root = os.path.join('E:', 'datasets', dataset, 'val')
    return train_root, val_root


def load_dataset(dataset_name, input_shape):
    train_root, val_root = get_dataset_roots('categorizer', dataset_name)
    dataset_classes = os.listdir(train_root)

    x_train_filenames = [os.path.join(train_root, dataset_class, filename)
                         for dataset_class in dataset_classes
                         for filename in os.listdir(os.path.join(train_root, dataset_class))]
    # y_train = np.zeros((len(x_train_filenames), len(dataset_classes)))
    x_train = np.zeros((len(x_train_filenames), input_shape[0], input_shape[1], input_shape[2]))
    y_train = np.zeros((len(x_train_filenames)))
    print('Load training_dataset:', len(x_train_filenames))
    for i, x_train_filename in tqdm(enumerate(x_train_filenames)):
        x_train[i] = read(x_train_filename, input_shape)
        y_train[i] = dataset_classes.index(os.path.split(os.path.split(x_train_filename)[0])[-1])
        # y_train[i, dataset_classes.index(os.path.split(os.path.split(x_train_filename)[0])[-1])] = 1

    x_val_filenames = [os.path.join(val_root, dataset_class, filename)
                       for dataset_class in dataset_classes
                       for filename in os.listdir(os.path.join(val_root, dataset_class))]
    # y_val = np.zeros((len(x_val_filenames), len(dataset_classes)))
    x_val = np.zeros((len(x_val_filenames), input_shape[0], input_shape[1], input_shape[2]))
    y_val = np.zeros((len(x_val_filenames)))
    print('Load validation_dataset:', len(x_val_filenames))
    for i, x_val_filename in tqdm(enumerate(x_val_filenames)):
        x_val[i] = read(x_val_filename, input_shape)
        y_val[i] = dataset_classes.index(os.path.split(os.path.split(x_val_filename)[0])[-1])
        # y_val[i, dataset_classes.index(os.path.split(os.path.split(x_val_filename)[0])[-1])] = 1
    return (x_train, y_train), (x_val, y_val), dataset_classes
