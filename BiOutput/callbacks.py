import os
import matplotlib.pyplot as plt

from keras.callbacks import Callback

from Rignak_Misc.path import get_local_file

HISTORY_CALLBACK_ROOT = os.path.join('_outputs', 'history')
EXAMPLE_CALLBACK_ROOT = os.path.join('_outputs', 'example')
CONFUSION_CALLBACK_ROOT = os.path.join('_outputs', 'confusion')


class ExampleCallback(Callback):
    def __init__(self, generator, root=EXAMPLE_CALLBACK_ROOT):
        super().__init__()
        self.root = root
        self.generator = generator

    def on_train_begin(self, logs=None):
        os.makedirs(os.path.join(self.root, self.model.name), exist_ok=True)
        self.on_epoch_end(0, logs=logs)

    def on_epoch_end(self, epoch, logs=None):
        example = next(self.generator)
        plot_example(example[0], self.model.predict(example[0]), self.model.labels, example[1])
        plt.savefig(os.path.join(self.root, self.model.name, f'{self.model.name}_{epoch}.png'))
        plt.savefig(os.path.join(self.root, f'{self.model.name}_current.png'))
        plt.close()


def plot_example(input_images, prediction, labels, groundtruth):
    n = input_images.shape[0]
    input_images = input_images[:, :, :, ::-1]
    prediction_images = prediction[1][:, :, :, ::-1]
    prediction_labels = prediction[0]
    groundtruth_images = groundtruth[1][:, :, :, ::-1]
    groundtruth_labels = groundtruth[0]

    plt.figure(figsize=(20, 10))
    for i, (input_image, prediction_image, prediction_label, groundtruth_image, groundtruth_label) in enumerate(zip(
            input_images, prediction_images, prediction_labels, groundtruth_images, groundtruth_labels)):
        if i != 0:
            tick_label = [' ' for label in labels]
        else:
            tick_label = labels

        plt.subplot(4, n, 1 + i)
        plt.imshow(input_image)

        plt.subplot(4, n, 1 + i + n)
        plt.imshow(prediction_image)

        plt.subplot(4, n, 1 + i + 2 * n)
        plt.imshow(groundtruth_image)

        plt.subplot(4, n, 1 + i + 3 * n)
        plt.barh(labels, prediction_label, tick_label=tick_label)
        plt.xlim(0, 1)
    plt.tight_layout()
