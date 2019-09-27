import numpy as np
import os
import sys
from PIL import Image

from Rignak_Misc.path import get_local_file
from Rignak_DeepLearning.callbacks import ConfusionCallback
from Rignak_DeepLearning.Categorizer.flat import import_model
from Rignak_DeepLearning.generator import fake_generator

INPUT_SHAPE = (128, 128, 1)
CONV_LAYERS = ((8, 1), (16, 1), (16, 2), (32, 3), (32, 3))
DENSE_LAYERS = (64, 64)

TRAINING_RATIO = 0.8
DATASET_ROOT = get_local_file(__file__, 'datasets')

BATCH_SIZE = 64
EPOCHS = 1000
NAME= "Speech"


def create_dataset(dataset, training_ratio=TRAINING_RATIO, dataset_root=DATASET_ROOT):
    folders = os.listdir(os.path.join(dataset_root, dataset))
    training_set, validation_set = [], []
    label_number = len(folders)
    for i, folder in enumerate(folders):
        folder = os.path.join(dataset_root, dataset, folder)
        filenames = os.listdir(folder)
        for j, filename in enumerate(filenames):
            with Image.open(os.path.join(folder, filename)) as image:
                input_ = np.array(image)[:, :, :1]
            groundtruth = np.zeros(label_number)
            groundtruth[i] = 1

            if j < training_ratio * len(filenames):
                training_set.append((input_, groundtruth))
            else:
                validation_set.append((input_, groundtruth))
    return np.array(training_set), np.array(validation_set), folders


def parse_input(argvs):
    dataset = argvs[1]
    return dataset


def train(training_set, validation_set, labels, input_shape=INPUT_SHAPE, conv_layers=CONV_LAYERS,
          dense_layers=DENSE_LAYERS, batch_size=BATCH_SIZE, epochs=EPOCHS, name=NAME):
    canals = validation_set.shape[-1]
    model = import_model(canals, input_shape=input_shape, conv_layers=conv_layers, dense_layers=dense_layers, name=name)

    model.fit(x=training_set[0], y=training_set[1], batch_size=batch_size, epochs=epochs,
              validation_data=validation_set, verbose=2,
              callbacks=[ConfusionCallback(fake_generator(training_set), labels)])


if __name__ == '__main__':
    training_set, validation_set, labels = create_dataset(parse_input(sys.argv))
    train(training_set, validation_set, labels)
