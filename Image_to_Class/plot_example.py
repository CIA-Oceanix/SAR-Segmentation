import matplotlib.pyplot as plt

from Rignak_Misc.plt import imshow

LIMIT = 8


def plot_example(example, prediction, labels, limit=LIMIT, denormalizer=None):
    input_, truth = example
    input_ = input_[:limit]
    n = input_.shape[0]
    plt.figure(figsize=(27, 12))
    plt.tight_layout()
    for i, (im, classes) in enumerate(zip(input_, prediction)):
        if i != 0:
            tick_label = [' ' for _ in labels]
        else:
            tick_label = labels
        plt.subplot(2, n, i + 1)
        imshow(im, denormalizer=denormalizer, vmax=1)

        plt.subplot(2, n, i + 1 + n)
        plt.barh(labels, truth[i], tick_label=tick_label, color='C1')
        plt.barh(labels, classes, tick_label=tick_label, color='C0')
        plt.xlim(0, 1)
        plt.yticks()
    plt.tight_layout()


def plot_regressor_distribution(examples, truths, predictions, attributes):
    plt.figure(figsize=(6 * (len(attributes) + 1), 4 * 4))

    for i, example in zip(range(3), examples):
        if i < len(example):
            plt.subplot(3, len(attributes) + 1, 1 + i * (len(attributes) + 1))
            imshow(example, vmax=1)
            if not i:
                plt.title('Example')

    for i, attribute in enumerate(attributes):
        plt.subplot(3, len(attributes) + 1, i + 2)
        plt.hist(predictions[:, i], bins=20, range=(-1, 1), density=True)
        plt.title(attribute)
        if not i:
            plt.ylabel('Predictions')

        plt.subplot(3, len(attributes) + 1, i + 2 + len(attributes) + 1)
        plt.hist(truths[:, i], bins=20, range=(-1, 1), density=True)
        if not i:
            plt.ylabel('Truth')

        plt.subplot(3, len(attributes) + 1, i + 2 + 2 * (len(attributes) + 1))
        plt.hist(abs(truths[:, i] - predictions[:, i]), bins=20, range=(0, 2), density=True)
        plt.ylim(0, 1)
        if not i:
            plt.ylabel('Absolute Error')

    plt.tight_layout()
