import os
import numpy as np
import matplotlib
import json

matplotlib.use('Agg')
import matplotlib.pyplot as plt

from tensorflow.keras.callbacks import Callback

from Rignak_Misc.plt import COLORS
from Rignak_Misc.path import get_local_file
from Rignak_DeepLearning.Image_to_Image.plot_example import plot_example as plot_autoencoder_example
from Rignak_DeepLearning.Image_to_Class.plot_example import plot_example as plot_categorizer_example
from Rignak_DeepLearning.Image_to_float.plot_example import plot_example as plot_regressor_example
from Rignak_DeepLearning.Image_to_Tag.plot_example import plot_example as plot_tagger_example

from Rignak_DeepLearning.Image_to_Class.confusion_matrix import compute_confusion_matrix, plot_confusion_matrix

CALLBACK_ROOT = get_local_file(__file__, os.path.join('_outputs'))


class HistoryCallback(Callback):
    """Callback generating a fitness plot in a file after each epoch"""

    def __init__(self, batch_size, training_steps, root=CALLBACK_ROOT):
        super().__init__()
        self.x = []
        self.logs = {}
        self.root = root
        self.batch_size = batch_size
        self.training_steps = training_steps

    def on_train_begin(self, logs=None):
        self.filename = os.path.join(self.root, self.model.name, 'history.png')
        os.makedirs(os.path.split(self.filename)[0], exist_ok=True)

    def on_epoch_end(self, epoch, logs={}):
        base_metrics = ('accuracy', 'val_accuracy', 'loss', 'val_loss')
        plt.ioff()
        self.x.append((epoch + 1) * self.batch_size * self.training_steps / 1000)

        for key, value in logs.items():
            if key not in self.logs: 
                self.logs[key] = []
            self.logs[key].append(value)
        accuracy_logs = {
            key:value 
            for key, value in self.logs.items() 
            if 'accuracy' in key
        }
        
        cols = 2 if accuracy_logs else 1
        plt.figure(figsize=(6 * cols, 6))

        if accuracy_logs:
            plt.subplot(1, cols, 2)
            for key, values in accuracy_logs.items():
                plt.plot(self.x, values, label=key)
                
            plt.xlabel('kimgs')
            plt.ylabel('Accuracy')
            plt.ylim(0, 1)
            plt.legend()
        
        plt.subplot(1, cols, 1)
        for key, values in self.logs.items():
            if key in accuracy_logs: continue
            plt.plot(self.x, values, label=key)
        plt.xlabel('kimgs')
        plt.yscale('log')
        plt.legend()

        plt.tight_layout()
        plt.savefig(self.filename)
        plt.close()



class AutoencoderExampleCallback(Callback):
    def __init__(self, generator, root=CALLBACK_ROOT):
        super().__init__()
        self.root = root
        self.generator = generator
        self.val_loss = None

    def on_train_begin(self, logs=None):
        self.filename = os.path.join(self.root, self.model.name, 'example.png')
        os.makedirs(os.path.split(self.filename)[0], exist_ok=True)
        os.makedirs(os.path.splitext(self.filename)[0], exist_ok=True)
        self.on_epoch_end(0)

    def on_epoch_end(self, epoch, logs={}):
        #if self.val_loss is not None and logs['val_loss'] > self.val_loss:
        #    return
        self.val_loss = logs.get('val_loss', 0)

        example = [[], [], []]
        while len(example[0]) < 8:
            batch_input, batch_output = next(self.generator)
            example[0] += list(batch_input[0] if isinstance(batch_input, list) else batch_input)
            example[1] += list(batch_output)
            example[2] += list(self.model.predict(batch_input))
        example[0] = np.array(example[0])
        example[1] = np.array(example[1])
        example[2] = np.array(example[2])

        plot_autoencoder_example(example[0], example[2], groundtruth=example[1], labels=self.model.callback_titles)
        plt.savefig(os.path.join(os.path.splitext(self.filename)[0], f'{epoch}.png'))
        plt.savefig(self.filename)
        plt.close()


class ClassificationExampleCallback(Callback):
    def __init__(self, generator, root=CALLBACK_ROOT):
        super().__init__()
        self.root = root
        self.generator = generator
        self.val_loss = None

    def on_train_begin(self, logs=None):
        self.filename = os.path.join(self.root, self.model.name, 'example.png')
        os.makedirs(os.path.split(self.filename)[0], exist_ok=True)
        os.makedirs(os.path.splitext(self.filename)[0], exist_ok=True)
        self.on_epoch_end(0)

    def on_epoch_end(self, epoch, logs={}):
        #if self.val_loss is not None and logs['val_loss'] > self.val_loss:
        #    return
        #self.val_loss = logs['val_loss']

        example = [[], [], []]
        while len(example[0]) < 8:
            batch_input, batch_output = next(self.generator)
            example[0] += list(batch_input[0] if isinstance(batch_input, list) else batch_input)
            example[1] += list(batch_output)
            example[2] += list(self.model.predict(batch_input))
        example[0] = np.array(example[0])
        example[1] = np.array(example[1])
        example[2] = np.array(example[2])

        plot_categorizer_example(example[:2], example[2], self.model.labels)
        plt.savefig(os.path.join(os.path.splitext(self.filename)[0], f'{epoch}.png'))
        plt.savefig(self.filename)
        plt.close()


class RegressorCallback(Callback):
    def __init__(self, generator, validation_steps, attributes, means, stds, root=CALLBACK_ROOT):
        super().__init__()
        self.root = root
        self.generator = generator
        self.validation_steps = validation_steps
        self.means = means
        self.stds = stds
        self.attributes = "".join(attributes) == 'RGB'
        self.val_loss = None

    def on_train_begin(self, logs=None):
        self.filename = os.path.join(self.root, self.model.name, 'example.png')
        os.makedirs(os.path.split(self.filename)[0], exist_ok=True)
        os.makedirs(os.path.splitext(self.filename)[0], exist_ok=True)

    def on_epoch_end(self, epoch, logs={}):
        if self.val_loss is not None and logs['val_loss'] > self.val_loss:
            return
        self.val_loss = logs['val_loss']
        examples, truths, predictions = [], [], []
        i_step = 0
        while (self.attributes == 'RGB' and len(examples) < 12) or \
                (self.attributes != 'RGB' and i_step < self.validation_steps):
            i_step += 1
            batch_input, batch_output = next(self.generator)
            examples += list(batch_input[0] if isinstance(batch_input, list) else batch_input)
            truths += list(batch_output)
            predictions += list(self.model.predict(batch_input))

        plot_regressor_example(examples, np.array(truths), np.array(predictions), self.model.labels,
                               self.means, self.stds)
        plt.savefig(os.path.join(os.path.splitext(self.filename)[0], f'{epoch}.png'))
        plt.savefig(self.filename)
        plt.close()


class ConfusionCallback(Callback):
    def __init__(self, generator, labels, root=CALLBACK_ROOT):
        super().__init__()
        self.root = root
        self.generator = generator
        self.labels = labels
        self.val_loss = None

    def on_train_begin(self, logs=None):
        self.filename = os.path.join(self.root, self.model.name, 'confusion.png')
        os.makedirs(os.path.split(self.filename)[0], exist_ok=True)
        os.makedirs(os.path.splitext(self.filename)[0], exist_ok=True)

    def on_epoch_end(self, epoch, logs={}):
        if self.val_loss is not None and logs['val_loss'] > self.val_loss:
            return
        self.val_loss = logs['val_loss']

        confusion_matrix = compute_confusion_matrix(self.model, self.generator, canals=len(self.labels))
        plot_confusion_matrix(confusion_matrix, labels=self.labels)

        plt.savefig(os.path.join(os.path.splitext(self.filename)[0], f'{epoch}.png'))
        plt.savefig(self.filename)
        plt.close()


class TaggerCallback(Callback):
    def __init__(self, generator, validation_steps, labels, root=CALLBACK_ROOT):
        super().__init__()
        self.root = root
        self.generator = generator
        self.labels = labels
        self.validation_steps = validation_steps
        self.val_loss = None

    def on_train_begin(self, logs=None):
        self.filename = os.path.join(self.root, self.model.name, 'example.png')
        os.makedirs(os.path.split(self.filename)[0], exist_ok=True)
        os.makedirs(os.path.splitext(self.filename)[0], exist_ok=True)

    def on_epoch_end(self, epoch, logs={}):
        if self.val_loss is not None and logs['val_loss'] > self.val_loss:
            return
        examples, truths, predictions = [], [], []
        for i_step in range(self.validation_steps):
            batch_input, batch_output = next(self.generator)
            examples += list(batch_input[0] if isinstance(batch_input, list) else batch_input)
            truths += list(batch_output)
            predictions += list(self.model.predict(batch_input))

        plot_tagger_example(examples, np.array(truths), np.array(predictions), self.model.labels)
        plt.savefig(os.path.join(os.path.splitext(self.filename)[0], f'{epoch}.png'))
        plt.savefig(self.filename)
        plt.close()


class SaveAttributes(Callback):
    def __init__(self, generator, config, labels=None, max_examples=4):
        super().__init__()
        self.generator = generator
        self.config = config
        self.saved_logs = []
        self.labels = labels
        self.max_examples = max_examples
        self.val_loss = None

    def on_train_begin(self, logs=None):
        self.filename = self.model.weight_filename
        os.makedirs(os.path.split(self.filename)[0], exist_ok=True)
        os.makedirs(os.path.splitext(self.filename)[0], exist_ok=True)

    def on_epoch_end(self, epoch, logs={}):
        self.saved_logs.append({k:float(v) for k, v in logs.items()})

        val_losses = [log['val_loss'] for log in self.saved_logs]
        dict_to_save = {"_logs": self.saved_logs, "_labels": self.labels, "_config": self.config}
        with open(self.filename + '.json', 'w') as file:
            json.dump(dict_to_save, file, sort_keys=True, indent=4)

        input_, groundtruth = next(self.generator)
        if len(input_) > self.max_examples:
            input_ = input_[:self.max_examples]
            groundtruth = groundtruth[:self.max_examples]
        if np.argmax(val_losses) == len(val_losses) - 1:
            self.input_ = [e.tolist() for e in input_] if isinstance(input_, list) else input_.tolist()
            self.groundtruth = groundtruth.tolist()
            self.output = self.model.predict(input_).tolist()
        return
        samples_to_save = {'input': self.input_,
                           "output": self.output,
                           "groundtruth": self.groundtruth}
        with open(self.filename + '_samples.json', 'w') as file:
            json.dump(samples_to_save, file, sort_keys=True, indent=4)
