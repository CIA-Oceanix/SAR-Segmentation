import sys
import numpy as np

from keras import backend as K
from keras.losses import mean_squared_error

CLASS_WEIGHTS = None
for arg in sys.argv:
    if arg.startswith('--CLASS_WEIGHTS'):
        CLASS_WEIGHTS = np.array(eval(arg.split('=')[-1]), dtype=float)
        CLASS_WEIGHTS = CLASS_WEIGHTS / sum(CLASS_WEIGHTS) * len(CLASS_WEIGHTS)


def weighted_binary_crossentropy(y_true, y_pred, smooth=0.0001, class_weights=CLASS_WEIGHTS):
    size = K.sum(y_pred) + K.sum(1 - y_pred)
    weights = 1 - (K.sum(y_true) / size)

    positive_loss = - weights * y_true * K.log(y_pred + smooth)
    negative_loss = - (1 - weights) * (1 - y_true) * K.log(1 - y_pred + smooth)

    loss = positive_loss + negative_loss
    if class_weights is not None:
        class_weights = K.variable(class_weights)
        loss = class_weights * loss
    return loss


def weighted_mse(y_true, y_pred, class_weights=CLASS_WEIGHTS):
    loss = (y_true - y_pred) ** 2
    if class_weights is not None:
        class_weights = K.variable(class_weights)
        loss = class_weights * loss
    return loss


def dice_coef_loss(y_true, y_pred, class_weights=CLASS_WEIGHTS):
    def dice_coef(y_true, y_pred, smooth=.00001):
        loss = 2 * y_true * y_pred / (y_true + y_pred + smooth)
        return loss

    loss = 1 - dice_coef(y_true, y_pred)
    if CLASS_WEIGHTS is not None:
        loss = loss * class_weights
    return loss


def nexrad_mse(y_true, y_pred):  # mse on the first three canals, mask from the fourth
    loss = (y_true[:, :, :, :3] - y_pred[:, :, :, :3]) ** 2
    loss = loss * y_true[:, :, :, 3:]  # loss not computed on the alpha channel
    return loss


def get_metrics(metric, names):
    def metric_function(metric, index):
        return lambda y_true, y_pred: metric(y_true, y_pred)[..., index]

    metrics = []
    for i, name in enumerate(names):
        metrics.append(metric_function(metric, i))
        while ' ' in name:
            name = name.replace(' ', '_')
        metrics[-1].__name__ = name

    return metrics


LOSS_TRANSLATION = {'WBCE': weighted_binary_crossentropy,
                    'DICE': dice_coef_loss,
                    "mse": mean_squared_error,
                    "wmse": weighted_mse,
                    "NEXRAD_MSE": nexrad_mse}
