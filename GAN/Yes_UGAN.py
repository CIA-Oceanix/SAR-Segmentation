import os
import numpy as np
import tqdm

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Lambda, concatenate, GlobalAveragePooling2D, Dense, Conv2D
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.applications import MobileNetV3Small
from tensorflow.keras.callbacks import ModelCheckpoint

from keras_radam.training import RAdamOptimizer
import tensorflow.keras.backend as K

from Rignak_DeepLearning.Image_to_Image.unet import import_model as import_unet_model
from Rignak_Misc.path import get_local_file
from Rignak_DeepLearning.config import get_config
from Rignak_DeepLearning.models import write_summary

ROOT = get_local_file(__file__, os.path.join('..', '_outputs'))
LOAD = False
LEARNING_RATE = 10 ** -4

CONFIG_KEY = 'segmenter'
CONFIG = get_config()[CONFIG_KEY]
DEFAULT_NAME = CONFIG.get('NAME', 'DEFAULT_MODEL_NAME')


def import_discriminator(config, input_layer, learning_rate=LEARNING_RATE):
    input_shape = config.get('INPUT_SHAPE')
    input_layer_resize = input_layer
    if input_shape[0] < 75 or input_shape[1] < 75:  # minimal input for InceptionV3
        input_layer_resize = Lambda(lambda x: tf.compat.v1.image.resize_bilinear(x, (75, 75)))(input_layer)

    if input_shape[-1] == 1:
        input_layer_tricanals = concatenate([input_layer_resize, input_layer_resize, input_layer_resize])
    elif input_shape[-1] == 3:
        input_layer_tricanals = input_layer_resize
    else:
        input_layer_tricanals = Conv2D(3, (1, 1))(input_layer_resize)

    base_model = InceptionV3(input_tensor=input_layer_tricanals, classes=1, include_top=False, weights='imagenet')
    #base_model = MobileNetV3Small(input_tensor=input_layer_tricanals, classes=1, include_top=False)
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1, activation='sigmoid')(x)
    discriminator = Model(input_layer, outputs=[x], name="discriminator")
    discriminator.summary()
    return discriminator


class Yes_UGAN():
    def __init__(self, root=ROOT, load=LOAD, learning_rate=LEARNING_RATE,
                 config=CONFIG, name=DEFAULT_NAME, skip=True, metrics=None, labels=None):
        self.args = (root, load, learning_rate, config, name, skip, metrics, labels)
        self.name = config.get('NAME')

        self.generator = import_unet_model(root=root,
                                           learning_rate=learning_rate, config=config, name=self.name, skip=skip,
                                           metrics=metrics, labels=labels)

        input_layer = Input(self.generator.input_shape[1:])
        generator_prediction = self.generator(input_layer)

        self.discriminator = import_discriminator(config, input_layer, learning_rate=learning_rate)
        self.discriminator.compile(loss="mse", optimizer=RAdamOptimizer(learning_rate), metrics=['accuracy', 'mae', 'binary_crossentropy'])
        
        self.discriminator.trainable = False
        #for layer in self.discriminator.layers[:-1]: layer.trainable=False
        
        discriminator_prediction = self.discriminator(generator_prediction)

        self.adversarial_autoencoder = Model(input_layer, [generator_prediction, discriminator_prediction],
                                             name=self.name)
        self.loss_weights = [0.999, 0.001]
        self.adversarial_autoencoder.compile(loss=['mse', 'mse'],
                                             loss_weights=self.loss_weights, optimizer=RAdamOptimizer(learning_rate))
                                             
        self.weight_filenames = [os.path.join(root, self.name, f"{prefix}.h5") for prefix in 'GDA']
        self.summary_filename = os.path.join(root, self.name, "model.txt")
        
        self.generator.weight_filename = self.weight_filenames[0]
        self.discriminator.weight_filename = self.weight_filenames[1]
        self.adversarial_autoencoder.weight_filename = self.weight_filenames[2]
        
        self.generator.summary_filename = self.summary_filename
        self.generator.__name = self.name
        self.adversarial_autoencoder.summary_filename = self.summary_filename
        write_summary(self.adversarial_autoencoder)
        
        #self.generator.load_weights(self.generator.weight_filename)
        #self.discriminator.load_weights(self.discriminator.weight_filename)

    def fit(self, x, validation_data, verbose, steps_per_epoch, validation_steps,
                      epochs, callbacks, initial_epoch):
        generator = x
        batch_size = len(next(x)[0])
        ones = np.ones((batch_size, 1))
        zeros = np.zeros((batch_size, 1))
        
        def process_batch(generator):
            generator_input, generator_truth = next(generator)
            generator_input = generator_input.astype('float32')*0
            generator_truth = generator_truth.astype('float32')
            
            import matplotlib.pyplot as plt
            # Train the discriminator
            if not batch_i % 4:
                generator_prediction = self.generator.predict(generator_input)
                p1 = self.discriminator.predict(generator_input)
                p2 = self.discriminator.predict(generator_prediction)
                
                plt.figure(figsize=(8,4))
                plt.subplot(1,2,1)
                plt.imshow(generator_input[0])
                plt.title(p1[0])
                plt.subplot(1,2,2)
                plt.imshow(generator_prediction[0])
                plt.title(p2[0])
                
                error = self.discriminator.train_on_batch(
                    np.concatenate((generator_prediction, generator_prediction), axis=0), 
                    np.concatenate((ones, zeros), axis=0), reset_metrics=True)

                plt.suptitle(error)
                plt.savefig(f'test/{epoch}-{batch_i}.png')
                plt.close()
                

            # Train the generator
            generator_loss = self.adversarial_autoencoder.train_on_batch([generator_input],  [generator_truth, ones])
            return generator_loss

        for callback in callbacks:
            callback.model = self.generator
            callback.on_train_begin(logs=None)

        for epoch in range(epochs): 
            pbar = tqdm.trange(steps_per_epoch, bar_format='')

            generator_losses = []
            discriminator_losses = []
            total_losses = []
            for batch_i in range(steps_per_epoch):
                total_loss, generator_loss, discriminator_loss = process_batch(generator)
                total_losses.append(float(total_loss))
                generator_losses.append(float(generator_loss))
                discriminator_losses.append(float(discriminator_loss))
                pbar.set_description(
                    f"Epoch {epoch}/{epochs} - Batch {batch_i + 1}/{steps_per_epoch} - "
                    f"[D_loss = {np.mean(discriminator_losses):.4f} "
                    f" E_loss = {np.mean(generator_losses):.4f}"
                )
                pbar.update(1)

            validation_generator_losses = []
            validation_discriminator_losses = []
            validation_discriminator_accuracies = []
            for batch_i in range(validation_steps):
                generator_input, generator_truth = next(validation_data)
                generator_prediction = self.generator.predict_on_batch(generator_input)

                discriminator_input = np.concatenate([generator_prediction, generator_truth], axis=0)
                discriminator_truth = np.concatenate([ones, zeros], axis=0)
                discriminator_prediction = self.discriminator.predict_on_batch(discriminator_input)
                
                validation_generator_loss = np.mean((generator_truth - generator_prediction) ** 2)
                validation_generator_losses.append(validation_generator_loss)

                validation_discriminator_loss = np.mean((discriminator_truth - discriminator_prediction) ** 2)
                validation_discriminator_losses.append(validation_discriminator_loss)
                validation_discriminator_accuracy = np.mean(np.where(np.abs(discriminator_prediction - discriminator_truth) > 0.5, 0, 1))
                validation_discriminator_accuracies.append(validation_discriminator_accuracy)

            logs = {
                "loss": float(np.mean(generator_losses) * self.loss_weights[0] + 
                              np.mean(discriminator_losses) * self.loss_weights[1]),
                "val_loss": float(np.mean(validation_generator_losses) * self.loss_weights[0] +
                                  np.mean(validation_discriminator_losses) * self.loss_weights[1]),
                "accuracy": float(np.mean(discriminator_prediction[:16])),
                "val_accuracy": float(np.mean(np.mean(discriminator_prediction[16:]))),
                "generator_loss": float(np.mean(generator_losses)),
                "val_generator_loss": float(np.mean(validation_generator_losses) * self.loss_weights[0]),
                "discriminator_loss": float(np.mean(discriminator_losses)),
                "val_discriminator_loss": float(np.mean(validation_discriminator_losses) * self.loss_weights[1])
            }

            for callback in callbacks:
                callback.model = self.generator
                callback.on_epoch_end(epoch=epoch, logs=logs)
                 
            self.generator.save(self.generator.weight_filename)
            self.discriminator.save(self.discriminator.weight_filename)
            

