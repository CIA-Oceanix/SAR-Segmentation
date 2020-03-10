import os
import numpy as np
import matplotlib
import json

matplotlib.use('Agg')
import matplotlib.pyplot as plt

from keras.callbacks import Callback

from Rignak_Misc.path import get_local_file
from Rignak_DeepLearning.Autoencoders.plot_example import plot_example as plot_autoencoder_example
from Rignak_DeepLearning.Categorizer.plot_example import plot_example as plot_categorizer_example
from Rignak_DeepLearning.Categorizer.confusion_matrix import compute_confusion_matrix, plot_confusion_matrix

HISTORY_CALLBACK_ROOT = get_local_file(__file__, os.path.join('_outputs', 'history'))
EXAMPLE_CALLBACK_ROOT = get_local_file(__file__, os.path.join('_outputs', 'example'))
CONFUSION_CALLBACK_ROOT = get_local_file(__file__, os.path.join('_outputs', 'confusion'))


class HistoryCallback(Callback):
    """Callback generating a fitness plot in a file after each epoch"""

    def __init__(self, root=HISTORY_CALLBACK_ROOT):
        super().__init__()
        self.x = []
        self.losses = []
        self.val_losses = []
        self.accuracy = [None]
        self.val_accuracy = [None]
        self.logs = []
        self.root = root

    def on_train_begin(self, logs=None):
        filename = os.path.join(self.root, f'{self.model.name}.png')
        os.makedirs(os.path.split(filename)[0], exist_ok=True)
        self.on_epoch_end(0, logs=logs)

    def on_epoch_end(self, epoch, logs={}):
        plt.ioff()
        self.logs.append(logs)
        self.x.append(epoch)
        self.losses.append(logs.get('loss'))
        self.val_losses.append(logs.get('val_loss'))

        if logs.get('acc'):
            self.accuracy.append(logs.get('acc'))
            self.val_accuracy.append(logs.get('val_acc'))

            plt.subplot(1, 2, 2)
            plt.plot(self.x, self.accuracy, label="Training")
            plt.plot(self.x, self.val_accuracy, label="Validation")
            plt.xlabel('Epoch')
            plt.ylabel('Accuracy')
            plt.ylim(0, 1)
            plt.legend()
            plt.subplot(1, 2, 1)

        plt.plot(self.x, self.losses, label="Training")
        plt.plot(self.x, self.val_losses, label="Validation")
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.yscale('log')
        plt.legend()

        plt.tight_layout()
        plt.savefig(os.path.join(self.root, f'{self.model.name}.png'))
        plt.close()


class AutoencoderExampleCallback(Callback):
    def __init__(self, generator, root=EXAMPLE_CALLBACK_ROOT, denormalizer=None):
        super().__init__()
        self.root = root
        self.generator = generator
        self.denormalizer = denormalizer

    def on_train_begin(self, logs=None):
        filename = os.path.join(self.root, self.model.name, f'{self.model.name}.png')
        os.makedirs(os.path.split(filename)[0], exist_ok=True)
        self.on_epoch_end(0, logs=logs)

    def on_epoch_end(self, epoch, logs={}):
        example = [[], [], []]
        while len(example[0]) < 8:
            next_ = next(self.generator)
            example[0] += list(next_[0])
            example[1] += list(next_[1])
            example[2] += list(self.model.predict(next_[0]))
        example[0] = np.array(example[0])
        example[1] = np.array(example[1])
        example[2] = np.array(example[2])

        plot_autoencoder_example(example[0], example[2], groundtruth=example[1], labels=self.model.callback_titles,
                                 denormalizer=self.denormalizer)
        plt.savefig(os.path.join(self.root, self.model.name, f'{os.path.split(self.model.name)[-1]}_{epoch}.png'))
        plt.savefig(os.path.join(self.root, f'{self.model.name}_current.png'))
        plt.close()


class ClassificationExampleCallback(Callback):
    def __init__(self, generator, root=EXAMPLE_CALLBACK_ROOT, denormalizer=None):
        super().__init__()
        self.root = root
        self.generator = generator
        self.denormalizer = denormalizer

    def on_train_begin(self, logs=None):
        filename = os.path.join(self.root, self.model.name, f'{self.model.name}.png')
        os.makedirs(os.path.split(filename)[0], exist_ok=True)
        self.on_epoch_end(0, logs=logs)

    def on_epoch_end(self, epoch, logs=None):
        example = [[], [], []]
        while len(example[0]) < 8:
            next_ = next(self.generator)
            example[0] += list(next_[0])
            example[1] += list(next_[1])
            example[2] += list(self.model.predict(next_[0]))
        example[0] = np.array(example[0])
        example[1] = np.array(example[1])
        example[2] = np.array(example[2])

        plot_categorizer_example(example[:2], example[2], self.model.labels, denormalizer=self.denormalizer)
        plt.savefig(os.path.join(self.root, self.model.name, f'{os.path.split(self.model.name)[-1]}_{epoch}.png'))
        plt.savefig(os.path.join(self.root, f'{self.model.name}_current.png'))
        plt.close()


class ConfusionCallback(Callback):
    def __init__(self, generator, labels, root=CONFUSION_CALLBACK_ROOT):
        super().__init__()
        self.root = root
        self.generator = generator
        self.labels = labels

    def on_train_begin(self, logs=None):
        filename = os.path.join(self.root, self.model.name, f'{self.model.name}.png')
        os.makedirs(os.path.split(filename)[0], exist_ok=True)

    def on_epoch_end(self, epoch, logs={}):
        confusion_matrix = compute_confusion_matrix(self.model, self.generator, canals=len(self.labels))
        plot_confusion_matrix(confusion_matrix, labels=self.labels)

        plt.savefig(os.path.join(self.root, self.model.name, f'{os.path.split(self.model.name)[-1]}_{epoch}.png'))
        plt.savefig(os.path.join(self.root, f'{self.model.name}_current.png'))
        plt.close()


class SaveAttributes(Callback):
    def __init__(self, generator, config, labels=None):
        super().__init__()
        self.generator = generator
        self.config = config
        self.saved_logs = []
        self.labels = labels

    def on_train_begin(self, logs=None):
        filename = self.model.weight_filename + '.json'
        os.makedirs(os.path.split(filename)[0], exist_ok=True)

    def on_epoch_end(self, epoch, logs=None):
        self.saved_logs.append(logs.copy())
        example = next(self.generator)

        val_losses = [log['val_loss'] for log in self.saved_logs]
        dict_to_save = {"_logs": self.saved_logs, "_labels": self.labels, "_config": self.config}
        if np.argmax(val_losses) == len(val_losses) - 1:
            self.input_ = example[0].tolist()
            self.groundtruth = example[1].tolist()
            self.output = self.model.predict(example[0]).tolist()
        dict_to_save['input'] = self.input_
        dict_to_save['output'] = self.output
        dict_to_save['groundtruth'] = self.groundtruth
        with open(self.model.weight_filename + '.json', 'w') as file:
            json.dump(dict_to_save, file, sort_keys=True, indent=4)
