import os
import numpy as np
import tqdm
import matplotlib.pyplot as plt

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Lambda, concatenate, GlobalAveragePooling2D, Dense, Conv2D
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.applications import ResNet50V2
from tensorflow.keras.callbacks import ModelCheckpoint

from tensorflow.keras import applications


from tensorflow.keras.optimizers import Adam
import tensorflow.keras.backend as K

from Rignak_DeepLearning.Image_to_Image.unet import import_model as import_unet_model, build_unet
from Rignak_Misc.path import get_local_file
from Rignak_DeepLearning.config import get_config
from Rignak_DeepLearning.models import write_summary
from Rignak_DeepLearning.loss import LOSS_TRANSLATION

ROOT = get_local_file(__file__, os.path.join('..', '_outputs'))
LOAD = False
LEARNING_RATE = 10 ** -4

CONFIG_KEY = 'segmenter'
CONFIG = get_config()[CONFIG_KEY]
DEFAULT_NAME = CONFIG.get('NAME', 'DEFAULT_MODEL_NAME')

    
def import_discriminator(config, learning_rate=LEARNING_RATE):
    args = {
        'input_shape': config['INPUT_SHAPE'], 
        "activation": 'relu',
        "batch_normalization": False,
        "conv_layers": config['CONV_LAYERS'],
        "skip": True,
        "last_activation": 'sigmoid',
        "output_shape": config['INPUT_SHAPE'],
        "central_shape": config['CONV_LAYERS'][-1],
        "resnet": False,
        "name": "discriminator",
    }
    discriminator = build_unet(**args)
    return discriminator


class Yes_UGAN():
    def __init__(self, root=ROOT, load=LOAD, learning_rate=LEARNING_RATE,
                 config=CONFIG, name=DEFAULT_NAME, skip=True, metrics=None, labels=None):
        self.args = (root, load, learning_rate, config, name, skip, metrics, labels)
        self.name = config.get('NAME')
        
        self.discriminator = import_discriminator(config, learning_rate=learning_rate)
        
        config['INPUT_SHAPE'] = list(config['INPUT_SHAPE'])
        config['INPUT_SHAPE'][-1] = config['INPUT_SHAPE'][-1] + 1
        self.generator = import_unet_model(root=root,
                                           learning_rate=learning_rate, config=config, name=self.name, skip=skip,
                                           metrics=metrics, labels=labels)

        input_layer = Input(self.generator.input_shape[1:])
        generator_prediction = self.generator(input_layer)

                                           
        self.discriminator.compile(loss="mse", optimizer=Adam(learning_rate), metrics=['accuracy', 'mae', 'binary_crossentropy'])
        self.discriminator.trainable = False
        
        discriminator_prediction = self.discriminator(generator_prediction)

        self.adversarial_autoencoder = Model(input_layer, [generator_prediction, discriminator_prediction], name=self.name)
        self.loss_weights = [5., 1.]
        self.adversarial_autoencoder.compile(loss=[LOSS_TRANSLATION[config.get('LOSS','mse')], 'mse'],
                                             loss_weights=self.loss_weights, optimizer=Adam(learning_rate))
                                             
        self.weight_filenames = [os.path.join(root, self.name, f"{prefix}.h5") for prefix in 'GDA']
        self.summary_filename = os.path.join(root, self.name, "model.txt")
        
        self.generator.weight_filename = self.weight_filenames[0]
        self.discriminator.weight_filename = self.weight_filenames[1]
        self.adversarial_autoencoder.weight_filename = self.weight_filenames[2]
        
        self.generator.summary_filename = self.summary_filename
        self.generator.__name = self.name
        self.adversarial_autoencoder.summary_filename = self.summary_filename
        write_summary(self.adversarial_autoencoder)
        
        self.adversarial_autoencoder.summary()
        
        #self.generator.load_weights(self.generator.weight_filename)
        #self.discriminator.load_weights(self.discriminator.weight_filename)

    def fit(self, x, validation_data, verbose, steps_per_epoch, validation_steps,
                      epochs, callbacks, initial_epoch):
        generator = x
        
        first_generator_input, first_generator_groundtruth = next(x)
        batch_size = first_generator_input.shape[0]
        output_shape = [batch_size] + list(self.discriminator.output_shape[1:-1]) + [1]
        
        ones = np.ones(output_shape)
        zeros = np.zeros(output_shape)
        
        def process_batch(generator):
            generator_input, generator_truth = next(generator)
            generator_input = generator_input.astype('float32')
            generator_truth = generator_truth.astype('float32')
            
            # Train the discriminator
            if not batch_i % 4:
                generator_prediction = self.generator.predict(generator_input)
                train_error_on_valid = self.discriminator.train_on_batch(generator_truth[:,:,:,[0]], ones + 0.01*K.random_normal(output_shape), reset_metrics=True)
                train_error_on_fakes = self.discriminator.train_on_batch(generator_prediction, zeros + 0.01*K.random_normal(output_shape), reset_metrics=True)
                    
            # Train the generator
            generator_loss = self.adversarial_autoencoder.train_on_batch([generator_input],  [generator_truth, ones])

            if not batch_i % 128:  # custum callback to check the D
                callback_filename = os.path.join('_outputs', self.name, 'D', f'{epoch}-{batch_i}.png')
                os.makedirs(os.path.split(callback_filename)[0], exist_ok=True)
                
                first_generator_prediction = self.generator.predict(first_generator_input)
                p1 = self.discriminator.predict(first_generator_groundtruth[:,:,:,[0]])
                p2 = self.discriminator.predict(first_generator_prediction)
                
                plt.figure(figsize=(16,14))
                for k in range(3):
                    plt.subplot(3,4,k*4+1)
                    plt.imshow(first_generator_input[k,:,:,0], cmap="gray")
                    plt.subplot(3,4,k*4+2)
                    plt.imshow(first_generator_prediction[k,:,:,0], cmap="jet", vmax=1, vmin=0)
                    plt.subplot(3,4,k*4+3)
                    plt.imshow(p1[k,:,:,0], cmap="jet", vmax=1, vmin=0)
                    plt.subplot(3,4,k*4+4)
                    plt.imshow(p2[k,:,:,0], cmap="jet", vmax=1, vmin=0)
                #plt.suptitle(f"Train Error:\n{train_error_on_valid}\n{train_error_on_fakes}")
                plt.savefig(callback_filename)
                plt.close()
                

            # Train the generator
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

                discriminator_input = np.concatenate([generator_prediction, generator_truth[:,:,:,[0]]], axis=0)
                discriminator_truth = np.concatenate([zeros, ones], axis=0)
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
                 
            self.generator.save_weights(self.generator.weight_filename)
            self.discriminator.save_weights(self.discriminator.weight_filename)
            
