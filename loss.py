import numpy as np
#import imutils
from functools import lru_cache

from keras.applications.vgg16 import VGG16
from keras.models import Model
from keras import backend as K


def dice_coef_loss(y_true, y_pred):
    def dice_coef(y_true, y_pred, smooth=.1):
        y_true_f = K.flatten(y_true)
        y_pred_f = K.flatten(y_pred)
        intersection = K.sum(y_true_f * y_pred_f)
        return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

    return 1 - dice_coef(y_true, y_pred)


def encode(latent_vector):
    if not isinstance(latent_vector, np.ndarray):
        latent_vector = np.array(latent_vector)
    encoded = latent_vector.tobytes(), latent_vector.shape, latent_vector.dtype
    return encoded


def decode(encoded):
    latent_vector, shape, data_type = encoded
    decoded = np.frombuffer(latent_vector, dtype=data_type).reshape(shape)
    if len(decoded.shape) <= 2 and decoded.shape[0] != 1:
        decoded = np.expand_dims(decoded, 0)
    return decoded


def get_perceptual_loss(input_size, layer_number=9, generative=None, features_only=False):
    @lru_cache(maxsize=64)
    def get_features_from_encoded_latent(encoded):
        input_batch = decode(encoded)
        input_batch = np.array(generative(input_batch))
        return get_features(input_batch)

    def get_features(input_batch):
        if len(input_batch.shape) == 3:
            input_batch = np.expand_dims(input_batch, 0)
        if input_batch.shape[-2] == input_size or input_batch.shape[-3] != input_size:
            input_batch = np.array([imutils.resize(image, input_size) for image in input_batch])
        features = perceptual_model.predict(input_batch)
        return features

    def perceptual_loss(input_batch, output_batch, args_are_features=False):
        if args_are_features:
            input_batch_features = input_batch
            output_batch_features = output_batch
        elif generative is not None:
            input_batch_features = get_features_from_encoded_latent(encode(input_batch))
            output_batch_features = get_features_from_encoded_latent(encode(output_batch))
        else:
            input_batch_features = get_features(input_batch)
            output_batch_features = get_features(output_batch)

        losses = []
        for input_features, output_features in zip(input_batch_features, output_batch_features):
            losses.append(((np.ravel(input_features) - np.ravel(output_features)) ** 2).mean() / 82890.0)
        return sum(losses)

    vgg16 = VGG16(include_top=False, input_shape=(input_size, input_size, 3))
    perceptual_model = Model(vgg16.input, vgg16.layers[layer_number].output)
    if features_only:
        return get_features_from_encoded_latent
    else:
        return perceptual_loss


def get_discriminator_loss(input_size, discriminative):
    input_size = 1024

    def discriminator_loss(input_batch):
        if not isinstance(input_batch, np.ndarray):
            input_batch = np.array(input_batch)
        if input_batch.shape[-2] == input_size or input_batch.shape[-3] != input_size:
            input_batch = np.array([imutils.resize(image, input_size) for image in input_batch])
        input_batch = np.transpose(input_batch, [0, 3, 2, 1])
        return -sum(sum(discriminative(input_batch, None)))

    return discriminator_loss


def get_space_roughness(generative, input_size, layers=(1, 2, 3, 4, 5, 6), batch_size=8, step=0.01):
    @lru_cache(maxsize=64)
    def space_roughness_from_encoded(encoded_input):
        input_batch = decode(encoded_input)
        loss = 0
        input_features = get_features(encode(input_batch))
        if len(input_batch.shape) == 1:
            input_batch = np.expand_dims(input_batch, axis=0)
        if len(input_batch.shape) == 3:
            input_batch = input_batch[0]

        for i in range(batch_size):
            neighbour = input_batch.copy()
            neighbour[[layers]] += (np.random.random((len(layers), neighbour.shape[-1])) - 0.5) * step
            output_features = get_features(encode(neighbour))
            loss += perceptual_loss(input_features, output_features, args_are_features=True)

        return loss / batch_size

    def space_roughness(input_batch, output_batch=None):
        return space_roughness_from_encoded(encode(input_batch))

    get_features = get_perceptual_loss(input_size, generative=generative, features_only=True)
    perceptual_loss = get_perceptual_loss(input_size, generative=generative)

    return space_roughness
