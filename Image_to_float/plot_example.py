import numpy as np
import matplotlib.pyplot as plt

from Rignak_Misc.plt import imshow


def plot_regressor_on_color(examples, truths, predictions, means, stds):
    plt.figure(figsize=(4 * 4, 3 * 4))

    for i, (example, truth, prediction) in enumerate(zip(examples[:12], truths, predictions)):
        plt.subplot(3, 4, i + 1)
        imshow(example, vmax=1)

        prediction = prediction * stds + means
        truth = truth * stds + means
        if means[0] > 1:
            prediction = prediction / 255
            truth = truth / 255

        plt.text(example.shape[0] / 2 * 1.05, -example.shape[1] / 12, '████', color=np.clip(prediction, 0, 1))
        plt.text(example.shape[0] / 2 * 1.05, -example.shape[1] / 35, '████', color=np.clip(truth, 0, 1))
        plt.title(f'\nPrediction:\nTruth:', fontdict={'horizontalalignment': 'right'})


def plot_regressor_distribution(examples, truths, predictions, attributes, means, stds):
    plt.figure(figsize=(6 * (len(attributes) + 1), 4 * 4))

    for i, example in zip(range(3), examples):
        if i < len(example):
            plt.subplot(3, len(attributes) + 1, 1 + i * (len(attributes) + 1))
            imshow(example, vmax=1)

            prediction = np.round(predictions[i] * stds + means, 2)
            truth = np.round(truths[i] * stds + means, 2)
            plt.title(f'Prediction: {prediction}\nTruth: {truth}', fontdict={'horizontalalignment': 'right'})

    for i, attribute in enumerate(attributes):
        mean = means[i]
        std = stds[i]

        plt.subplot(3, len(attributes) + 1, i + 2)

        plt.hist(predictions[:, i] * std + mean, bins=20, density=True)
        plt.title(attribute)
        if not i:
            plt.ylabel('Predictions')

        plt.subplot(3, len(attributes) + 1, i + 2 + len(attributes) + 1)
        plt.hist(truths[:, i] * std + mean, bins=20, density=True)
        if not i:
            plt.ylabel('Truth')

        plt.subplot(3, len(attributes) + 1, i + 2 + 2 * (len(attributes) + 1))
        plt.hist(abs(truths[:, i] - predictions[:, i]), bins=20, density=True)
        if not i:
            plt.ylabel('Normalized Absolute Error')


def plot_example(examples, truths, predictions, attributes, means, stds):
    means = np.array(means)
    stds = np.array(stds)
    if "".join(attributes) == 'RGB':
        plot_regressor_on_color(examples, truths, predictions, means, stds)
    else:
        plot_regressor_distribution(examples, truths, predictions, attributes, means, stds)
    plt.tight_layout()
