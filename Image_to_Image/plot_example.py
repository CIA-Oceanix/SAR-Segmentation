import matplotlib.pyplot as plt
import numpy as np

from Rignak_Misc.plt import imshow, COLORMAPS

MAX_THUMBS = 8

import matplotlib
cmap = matplotlib.cm.jet
cmap.set_bad('grey',1)


def plot_less_than_three_canals(input_, prediction, groundtruth, labels):
    plt.figure(figsize=(max(256, input_.shape[2]) / 100 * 9, max(256, input_.shape[1]) / 100 * 3.5))
    if groundtruth is None:
        line_number = 2
    else:
        line_number = 3
    col_number = input_.shape[0]

    for i, (input_thumb, pred_thumb, truth_thumb) in enumerate(zip(input_, prediction, groundtruth)):
        if len(truth_thumb.shape) == 3 and truth_thumb.shape[-1] == 4:  # we assume it's an alpha canal
            pred_thumb[truth_thumb[:, :, 3] < 0.5] = 0
            input_thumb[truth_thumb[:, :, 3] < 0.5] = 0
            truth_thumb[truth_thumb[:, :, 3] < 0.5] = 0
            
        plt.subplot(line_number, col_number, 1 + i)
        imshow(input_thumb, vmin=0, vmax=255)
        plt.title(f'{input_thumb.max()}')
        if not i and labels is not None:
            plt.title(labels[0])

        plt.subplot(line_number, col_number, i + 1 + col_number)
        imshow(pred_thumb, vmin=0, vmax=np.nanmax(groundtruth), cmap=cmap, threshold = 'auto')
        plt.title(f'{pred_thumb.max()}')
        if not i and labels is not None:
            plt.title(labels[1])

        if line_number == 3:
            plt.subplot(line_number, col_number, i + 1 + col_number * 2)
            imshow(truth_thumb[:, :, :min(truth_thumb.shape[-1], 3)], vmin=0, vmax=np.nanmax(groundtruth), cmap=cmap)
            plt.title(f'{truth_thumb.max()}')
            if not i and labels is not None:
                plt.title(labels[2]) 

def plot_more_than_three_canals(input_, prediction, groundtruth, labels):
    plt.figure(figsize=(40, 16))
    line_number = input_.shape[0]
    col_number = groundtruth.shape[-1] + 3
    for i, (input_thumb, pred_thumb, truth_thumb) in enumerate(zip(input_, prediction, groundtruth)):
        plt.subplot(line_number, col_number, i * col_number + 1)
        imshow(input_thumb, cmap='gray')
        if not i and labels is not None:
            plt.title(labels[0])

        plt.subplot(line_number, col_number, i * col_number + 2)
        imshow(pred_thumb)
        if not i and labels is not None:
            plt.title(labels[1])

        plt.subplot(line_number, col_number, i * col_number + 3)
        imshow(truth_thumb)
        if not i and labels is not None:
            plt.title(labels[2])

        for canal in range(pred_thumb.shape[2]):
            plt.subplot(line_number, col_number, i * col_number + canal + 4)
            imshow(pred_thumb[:, :, canal], cmap=COLORMAPS[canal])
            plt.colorbar()
            if not i and labels is not None and len(labels) > 3 + canal:
                plt.title(f"Prediction: {labels[3 + canal]}")


def plot_example(input_, prediction, groundtruth=None, max_thumbs=MAX_THUMBS, labels=None):
    input_ = input_[:max_thumbs]
    if len(input_.shape) == 4 and input_.shape[-1] == 2: input_ = input_[:,:,:,[0]] 

    if len(prediction.shape) == 4 and prediction.shape[-1] > 4:
        plot_more_than_three_canals(input_, prediction, groundtruth, labels)
    else:
        plot_less_than_three_canals(input_, prediction, groundtruth, labels)
    plt.tight_layout()
