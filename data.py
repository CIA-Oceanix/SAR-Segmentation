import os
import sys
import numpy as np
from PIL import Image
from skimage.transform import resize
from tqdm import tqdm
import functools
import PIL.Image

from keras.preprocessing import image as image_utils

from Rignak_Misc.path import convert_link

CANALS = 3

print('------------')
print(sys.argv)
print('------------')


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
    if filename.endswith('.lnk'):
        filename = convert_link(filename)
    try:
        if os.path.splitext(filename)[-1] == '.npy':
            with open(filename, 'rb') as numpy_filename:
                im = np.load(numpy_filename, allow_pickle=True)
        else:
            with PIL.Image.open(filename) as im:
                im = np.array(im) / 255
                if im.max() > 255:
                    im = im / 255
        if input_shape is not None and (im.shape[0] != input_shape[0] or im.shape[1] != input_shape[1]):
            im = resize(im, input_shape[:2], anti_aliasing=True)
        if len(im.shape) == 2:
            im = np.expand_dims(im, axis=-1)
        if im.shape[-1] == 1 and input_shape[-1] == 3:
            im = im[:, :, [0, 0, 0]]
        elif im.shape[-1] == 3 and input_shape[-1] == 1:
            im = np.expand_dims(np.mean(im, axis=-1), axis=-1)
    except Exception as e:
        print('exists:', os.path.exists(filename))
        print(f"Error {e} when reading {filename}, will return empty image")
        return np.zeros(input_shape)
    return im


if '--CACHE=True' in sys.argv:
    print('Be careful: you are using the RAM to store the dataset')
    hidden_read = read


    @functools.lru_cache(maxsize=50000)
    def cached_read(filename, input_shape=None):
        return hidden_read(filename, input_shape)


    read = cached_read


def get_dataset_roots(dataset='.', root='E:\\\\datasets'):
    train_root = os.path.join(root, dataset, 'train')
    val_root = os.path.join(root, dataset, 'val')

    # train_root = os.path.join('..', '..', '..', 'data', dataset)
    # val_root = os.path.join('..', '..', '..', 'data', dataset)
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
