import sys
import numpy as np

from keras import backend as K
from keras.losses import mean_squared_error


def siamese_loss(y_true, y_pred):
    return mean_squared_error(y_true[:, 2], y_pred)
