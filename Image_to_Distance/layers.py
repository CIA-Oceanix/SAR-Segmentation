import keras
import sys

import tensorflow as tf
from keras import backend as K
from keras.layers import Conv2D, MaxPooling2D, Dense, Input
from keras.models import Model, Sequential
from keras.engine import InputSpec, Layer
from keras import regularizers
from keras.optimizers import SGD, Adam
from keras.utils.conv_utils import conv_output_length
from keras import activations
import numpy as np


class CorrLayer(Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def build(self, inputs):
        super(CorrLayer, self).build(inputs)  # Be sure to call this at the end

    def call(self, inputs, **kwargs):
        def _corr(search_img, filter):
            x = tf.expand_dims(search_img, 0)
            f = tf.expand_dims(filter, -1)
            # use the feature map as kernel for the depthwise conv2d of tensorflow
            return tf.nn.depthwise_conv2d(input=x, filter=f, strides=[1, 1, 1, 1], padding='SAME')

        # Iteration over each batch
        out = tf.map_fn(
            lambda filter_simg: _corr(filter_simg[0], filter_simg[1]),
            elems=inputs,
            dtype=inputs[0].dtype
        )
        return tf.squeeze(out, [1])

    def compute_output_shape(self, input_shape):
        return input_shape
