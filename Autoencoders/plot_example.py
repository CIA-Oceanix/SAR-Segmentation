import matplotlib.pyplot as plt

from Rignak_Misc.plt import imshow, COLORMAPS

MAX_THUMBS = 8


def plot_less_than_three_canals(input_, prediction, groundtruth, labels, denormalizer=None):
    plt.figure(figsize=(20, 8))
    if groundtruth is None:
        line_number = 2
    else:
        line_number = 3
    col_number = input_.shape[0]

    for i, im in enumerate(input_):
        plt.subplot(line_number, col_number, 1 + i)
        imshow(input_[i], denormalizer=denormalizer, vmin=0, vmax=255)
        if not i and labels is not None:
            plt.title(labels[0])

        plt.subplot(line_number, col_number, i + 1 + col_number)
        imshow(prediction[i], denormalizer=denormalizer, vmin=0, vmax=255)
        if not i and labels is not None:
            plt.title(labels[1])

        if line_number == 3:
            plt.subplot(line_number, col_number, i + 1 + col_number * 2)
            imshow(groundtruth[i], denormalizer=denormalizer, vmin=0, vmax=255)
            if not i and labels is not None:
                plt.title(labels[2])


def plot_more_than_three_canals(input_, prediction, groundtruth, labels, denormalizer=None):
    plt.figure(figsize=(40, 16))
    line_number = input_.shape[0]
    col_number = groundtruth.shape[-1] + 3
    for i, (input_thumb, pred_thumb, truth_thumb) in enumerate(zip(input_, prediction, groundtruth)):
        plt.subplot(line_number, col_number, i * col_number + 1)
        imshow(input_thumb, cmap='gray', denormalizer=denormalizer)
        if not i and labels is not None:
            plt.title(labels[0])

        plt.subplot(line_number, col_number, i * col_number + 2)
        imshow(pred_thumb, denormalizer=denormalizer)
        if not i and labels is not None:
            plt.title(labels[1])

        plt.subplot(line_number, col_number, i * col_number + 3)
        imshow(truth_thumb, denormalizer=denormalizer)
        if not i and labels is not None:
            plt.title(labels[2])

        for canal in range(pred_thumb.shape[2]):
            plt.subplot(line_number, col_number, i * col_number + canal + 4)
            imshow(pred_thumb[:, :, canal], cmap=COLORMAPS[canal], denormalizer=denormalizer)
            plt.colorbar()
            if not i and labels is not None:
                plt.title(f"Prediction: {labels[2 + canal]}")


def plot_example(input_, prediction, groundtruth=None, max_thumbs=MAX_THUMBS, labels=None, denormalizer=None):
    input_ = input_[:max_thumbs]

    if len(prediction.shape) == 4 and prediction.shape[-1] > 3:
        plot_more_than_three_canals(input_, prediction, groundtruth, labels, denormalizer=denormalizer)
    else:
        plot_less_than_three_canals(input_, prediction, groundtruth, labels, denormalizer=denormalizer)
    plt.tight_layout()
