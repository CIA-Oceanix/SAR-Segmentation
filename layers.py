from keras.layers import Layer
from keras import backend as K


# Input b and g should be 1x1xC
class PixelNormLayer(Layer):
    def __init__(self, axis=-1, epsilon=1e-3, **kwargs):
        super().__init__(**kwargs)
        self.axis = axis
        self.epsilon = epsilon

    def build(self, input_shape):
        dim = input_shape[self.axis]
        if dim is None:
            raise ValueError(f'Axis {self.axis} of input tensor should have a defined dimension '
                             f'but the layer received an input with shape {input_shape}.')

        super().build(input_shape)

    def call(self, inputs, training=None):
        input_shape = K.int_shape(inputs)
        normed = inputs / K.expand_dims(K.sqrt(K.sum(inputs, axis=-1) / input_shape[-1] + self.epsilon), axis=-1)
        return normed

    def get_config(self):
        config = {'axis': self.axis, 'epsilon': self.epsilon}
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def compute_output_shape(self, input_shape):
        return input_shape
