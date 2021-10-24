import os

import matplotlib
import numpy as np
from Rignak_DeepLearning.Image_to_Class.confusion_matrix import plot_confusion_matrix
from Rignak_Misc.plt import imshow

matplotlib.use('Agg')
import matplotlib.pyplot as plt

from tensorflow.keras.callbacks import Callback

CALLBACK_ROOT = '_outputs'


class DistanceCallback(Callback):
    def __init__(self, generator, validation_steps, labels, root=CALLBACK_ROOT):
        super().__init__()
        self.root = root
        self.generator = generator
        self.labels = labels
        self.validation_steps = validation_steps

    def on_epoch_end(self, epoch, logs={}):
        normalized_confusion, count, batch_input, predictions = compute_confusion_matrix(self.model, self.generator,
                                                                                         len(self.labels),
                                                                                         self.validation_steps)

        plot_confusion_matrix(normalized_confusion, labels=self.labels)
        plt.savefig(os.path.join(os.path.splitext(self.filename)[0], f'{epoch}.png'))
        plt.savefig(self.filename)
        plt.close()

        plot_confusion_matrix(count, labels=self.labels, vmax=count.max())
        plt.savefig(os.path.join(os.path.splitext(self.filename)[0], f'{epoch}_count.png'))
        plt.close()

        plot_example(
            np.concatenate((batch_input[0][:3], batch_input[0][-3:])),
            np.concatenate((batch_input[1][:3], batch_input[1][-3:])),
            labels=list(predictions[:3])+list(predictions[-3:])
        )
        plt.savefig(os.path.join(os.path.splitext(self.filename)[0], f'example.png'))
        plt.close()

    def on_train_begin(self, logs=None):
        self.filename = os.path.join(self.root, self.model.name, 'confusion.png')
        os.makedirs(os.path.split(self.filename)[0], exist_ok=True)
        os.makedirs(os.path.splitext(self.filename)[0], exist_ok=True)
        self.on_epoch_end(0)


def compute_confusion_matrix(model, generator, n_labels, validation_steps):
    confusion_matrix = np.zeros((len(model.labels), len(model.labels), 2))

    for _ in range(validation_steps):
        batch_input, batch_output = next(generator)
        batch_input = [np.array(batch_input[0]), np.array(batch_input[1])]
        predictions = model.predict(batch_input)
        for prediction, (label_index_1, label_index_2, truth) in zip(predictions, batch_output):
            confusion_matrix[int(label_index_1), int(label_index_2), 0] += int(prediction > 0.5)
            confusion_matrix[int(label_index_1), int(label_index_2), 1] += 1
    return confusion_matrix[:, :, 0] / confusion_matrix[:, :, 1], confusion_matrix[:, :, 1], batch_input, predictions


def plot_example(inputs_1, inputs_2, labels):
    plt.figure(figsize=(max(256, inputs_1[0].shape[2]) / 100 * 9, max(256, inputs_1[0].shape[1]) / 100 * 3.5))
    line_number = 2
    col_number = len(inputs_1)

    for i, (input_1, input_2) in enumerate(zip(inputs_1, inputs_2)):
        plt.subplot(line_number, col_number, 1 + i)
        imshow(input_1, vmin=0, vmax=255)
        plt.title(labels[i])

        plt.subplot(line_number, col_number, i + 1 + col_number)
        imshow(input_2, vmin=0, vmax=255)
