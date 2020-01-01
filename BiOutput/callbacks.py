import os
import matplotlib.pyplot as plt

from keras.callbacks import Callback

HISTORY_CALLBACK_ROOT = os.path.join('_outputs', 'history')
EXAMPLE_CALLBACK_ROOT = os.path.join('_outputs', 'example')
CONFUSION_CALLBACK_ROOT = os.path.join('_outputs', 'confusion')


class HistoryCallback(Callback):
    """Callback generating a fitness plot in a file after each epoch"""

    def __init__(self, root=HISTORY_CALLBACK_ROOT):
        super().__init__()
        self.x = []
        self.losses = []
        self.val_losses = []
        self.accuracy = []
        self.val_accuracy = []
        self.categorizer_loss = []
        self.decoder_loss = []
        self.val_categorizer_loss = []
        self.val_decoder_loss = []
        self.logs = []
        self.root = root

    def on_train_begin(self, logs=None):
        self.on_epoch_end(0, logs=logs)

    def on_epoch_end(self, epoch, logs={}):
        plt.ioff()
        self.logs.append(logs)
        self.x.append(epoch)
        self.losses.append(logs.get('loss'))
        self.val_losses.append(logs.get('val_loss'))
        self.categorizer_loss.append(logs.get('categorizer_block_loss'))
        self.decoder_loss.append(logs.get('decoder_block_loss'))
        self.val_categorizer_loss.append(logs.get('val_categorizer_block_loss'))
        self.val_decoder_loss.append(logs.get('val_decoder_block_loss'))

        self.accuracy.append(logs.get('categorizer_block_acc'))
        self.val_accuracy.append(logs.get('val_categorizer_block_acc'))

        plt.subplot(1, 2, 2)
        plt.plot(self.x, self.accuracy, label="Training")
        plt.plot(self.x, self.val_accuracy, label="Validation")
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.ylim(0, 1)
        plt.legend()

        plt.subplot(1, 2, 1)
        plt.plot(self.x, self.losses, label="Training", color='deepskyblue')
        plt.plot(self.x, self.categorizer_loss, label='Training (categorizer)', color='lightcoral')
        plt.plot(self.x, self.val_categorizer_loss, label='Validation (categorizer)', color='red')
        plt.plot(self.x, self.val_losses, label="Validation", color='blue')
        plt.plot(self.x, self.decoder_loss, label='Training (decoder)', color='greenyellow')
        plt.plot(self.x, self.val_decoder_loss, label='Validation (decoder)', color='green')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.yscale('log')
        plt.legend()

        plt.tight_layout()
        plt.savefig(os.path.join(self.root, f'{self.model.name}.png'))
        plt.close()


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
    input_images = input_images
    prediction_images = prediction[1]
    prediction_labels = prediction[0]
    groundtruth_images = groundtruth[1]
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
        plt.barh(labels, groundtruth_label, tick_label=tick_label, color='C1')
        plt.barh(labels, prediction_label, tick_label=tick_label, color='C0')
        plt.xlim(0, 1)
    plt.tight_layout()
