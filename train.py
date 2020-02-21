import os, sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
for arg in sys.argv:
    if arg.startswith('--run_on_gpu='):
        os.environ['CUDA_VISIBLE_DEVICES'] = arg[-1]

import fire

import deprecation_warnings

deprecation_warnings.filter_warnings()

from keras.callbacks import ModelCheckpoint

from data import get_dataset_roots
from Rignak_DeepLearning.normalization import NORMALIZATION_FUNCTIONS
from Rignak_DeepLearning.noise import NOISE_FUNCTIONS
from Rignak_DeepLearning.callbacks import HistoryCallback, AutoencoderExampleCallback, ConfusionCallback, \
    ClassificationExampleCallback
from Rignak_DeepLearning.Autoencoders.flat import import_model as import_flat_model
from Rignak_DeepLearning.Autoencoders.unet import import_model as import_unet_model
from Rignak_DeepLearning.Categorizer.flat import import_model as import_categorizer
from Rignak_DeepLearning.Categorizer.Inception import import_model_v3 as InceptionV3
from Rignak_DeepLearning.BiOutput.flat import import_model as import_bimode
from Rignak_DeepLearning.BiOutput.multiscale_autoencoder import import_model as import_multiscale_bimode
from Rignak_DeepLearning.BiOutput.generator import generator as bimode_generator, \
    normalize_generator as bimode_normalize, augment_generator as bimode_augment
from Rignak_DeepLearning.BiOutput.callbacks import ExampleCallback as BimodeExampleCallback
from Rignak_DeepLearning.BiOutput.callbacks import HistoryCallback as BimodeHistoryCallback
from Rignak_DeepLearning.Categorization2Segmentation.heatmap_model import import_model as import_heatmap_model
from Rignak_DeepLearning.Categorization2Segmentation.generator import heatmap_generator
from Rignak_DeepLearning.generator import autoencoder_generator, categorizer_generator, saliency_generator, \
    thumbnail_generator as thumb_generator, normalize_generator, augment_generator, regressor_generator, \
    rotsym_augmentor
from Rignak_DeepLearning.StyleGan.callbacks import GanRegressorExampleCallback
from Rignak_DeepLearning.config import get_config

"""
>>> python train.py autoencoder fav-rignak 
>>> python train.py saliency open_eyes
>>> python train.py saliency eye_color
>>> python train.py categorizer waifu
>>> python train.py mnist mnist batch_size=256
>>> python train.py style_transfer colorization 
>>> python train.py bimode waifu 
>>> python train.py inceptionV3 chen --NORMALIZATION=fourier --NAME=TenGeoP_fourier --INPUT_SHAPE="(256,256,3) --epochs=15"
>>> python train.py inceptionV3 chen --NORMALIZATION=fourier_only --NAME=TenGeoP_fourier_only --INPUT_SHAPE="(256,256,1) --epochs=15"

python train.py inceptionV3 waifu2latent --NOISE=[contrast,uniform] --NORMALIZATION=intensity --IMAGENET=transfer --NAME=waifu2latent_512 --INPUT_SHAPE="(512,512,3)" --batch_size=4
python train.py inceptionV3 waifu_faces --NOISE=[contrast,uniform] --NORMALIZATION=intensity --IMAGENET=transfer --NAME=waifu_faces2latent_256 --INPUT_SHAPE="(256,256,3)" --batch_size=4

python train.py inceptionV3 --NOISE=[categorization] --NORMALIZATION=intensity --IMAGENET=transfer --NAME=TenGeoP-SARwv_512 --INPUT_SHAPE="(512,512,1)" --batch_size=4 --epochs=25
python train.py heatmap "TenGeoP-SARwv_heatmap/Atmospheric Front" --NORMALIZATION=intensity --NAME=TENGEOP-SARwv_icebergs_heatmap --INPUT_SHAPE="(512,512,1)" --OUTPUT_SHAPE="(32, 32, 1)" --batch_size=4 --INPUT_LABEL=SAR, --OUTPUT_LABEL="HEAT"
"""

BATCH_SIZE = 16
TRAINING_STEPS = 2000
VALIDATION_STEPS = 500
EPOCHS = 1000
INITIAL_EPOCH = 0

DEFAULT_INPUT_SHAPE = (256, 256, 3)
DEFAULT_SCALING = 1


def get_generators(config, task, dataset, batch_size, default_input_shape=DEFAULT_INPUT_SHAPE,
                   default_scaling=DEFAULT_SCALING):
    def get_saliency_generators():
        train_generator = saliency_generator(train_folder, input_shape=input_shape, batch_size=batch_size)
        val_generator = saliency_generator(val_folder, input_shape=input_shape, batch_size=batch_size)
        callback_generator = saliency_generator(val_folder, input_shape=input_shape, batch_size=batch_size)
        return train_generator, val_generator, callback_generator, train_folder

    def get_autoencoder_generators():
        train_generator = autoencoder_generator(train_folder, input_shape=input_shape, batch_size=batch_size)
        val_generator = autoencoder_generator(val_folder, input_shape=input_shape, batch_size=batch_size)
        callback_generator = autoencoder_generator(val_folder, input_shape=input_shape, batch_size=batch_size)
        return train_generator, val_generator, callback_generator, train_folder

    def get_categorizer_generators():
        train_generator = categorizer_generator(train_folder, input_shape=input_shape, batch_size=batch_size)
        val_generator = categorizer_generator(val_folder, input_shape=input_shape, batch_size=batch_size)
        callback_generator = categorizer_generator(val_folder, input_shape=input_shape, batch_size=batch_size)
        return train_generator, val_generator, callback_generator, train_folder

    def get_style_transfer_generators():
        train_generator = thumb_generator(train_folder, input_shape=input_shape, batch_size=batch_size, scaling=scaling)
        val_generator = thumb_generator(val_folder, input_shape=input_shape, batch_size=batch_size, scaling=scaling)
        callback_gene = thumb_generator(val_folder, input_shape=input_shape, batch_size=batch_size, scaling=scaling)
        return train_generator, val_generator, callback_gene, train_folder

    def get_bimode_generators():
        train_generator = bimode_generator(train_folder, input_shape=input_shape, batch_size=batch_size)
        val_generator = bimode_generator(val_folder, input_shape=input_shape, batch_size=batch_size)
        callback_generator = bimode_generator(val_folder, input_shape=input_shape, batch_size=batch_size)
        return train_generator, val_generator, callback_generator, train_folder

    def get_regressor_generators():
        train_generator = regressor_generator(train_folder, input_shape=input_shape, batch_size=batch_size)
        val_generator = regressor_generator(val_folder, input_shape=input_shape, batch_size=batch_size)
        callback_generator = regressor_generator(val_folder, input_shape=input_shape, batch_size=batch_size)
        return train_generator, val_generator, callback_generator, train_folder

    def get_heatmap_generators():
        output_shape = config[task].get('OUTPUT_SHAPE', input_shape)
        input_label = config[task].get('INPUT_LABEL', 'input')
        output_label = config[task].get('OUTPUT_LABEL', 'output')
        train_generator = heatmap_generator(train_folder, input_shape=input_shape, output_shape=output_shape,
                                            batch_size=batch_size, input_label=input_label, output_label=output_label)
        val_generator = heatmap_generator(val_folder, input_shape=input_shape, output_shape=output_shape,
                                          batch_size=batch_size, input_label=input_label, output_label=output_label)
        callback_generator = heatmap_generator(val_folder, input_shape=input_shape, output_shape=output_shape,
                                               batch_size=batch_size, input_label=input_label,
                                               output_label=output_label)
        return train_generator, val_generator, callback_generator, train_folder

    print('CREATING THE GENERATORS')
    input_shape = config[task].get('INPUT_SHAPE', default_input_shape)
    scaling = config[task].get('SCALING', default_scaling)
    train_folder, val_folder = get_dataset_roots(task, dataset=dataset)

    functions = {"saliency": get_saliency_generators,
                 "autoencoder": get_autoencoder_generators,
                 "heatmap": get_heatmap_generators,
                 "flat_autoencoder": get_autoencoder_generators,
                 "categorizer": get_categorizer_generators,
                 "inceptionV3": get_categorizer_generators,
                 "style_transfer": get_style_transfer_generators,
                 "bimode": get_bimode_generators,
                 "multiscale_bimode": get_bimode_generators,
                 "regressor": get_regressor_generators,
                 }
    return functions[task]()


def get_data_augmentation(config, task, train_generator, val_generator, callback_generator):
    def get_im2im_data_augmentation():
        new_train_generator = normalize_generator(
            augment_generator(train_generator, noise_function=noise_function, apply_on_output=True),
            normalizer, apply_on_output=True)
        new_val_generator = normalize_generator(
            augment_generator(val_generator, noise_function=noise_function, apply_on_output=True),
            normalizer, apply_on_output=True)
        new_callback_generator = normalize_generator(
            augment_generator(callback_generator, noise_function=noise_function, apply_on_output=True),
            normalizer, apply_on_output=True)
        return new_train_generator, new_val_generator, new_callback_generator

    def get_bimode_augmentation():
        # new_train_generator = bimode_normalize(
        #     bimode_augment(train_generator, noise_function=noise_function, apply_on_output=True),
        #     normalizer, apply_on_output=True)
        # new_val_generator = bimode_normalize(
        #     bimode_augment(val_generator, noise_function=noise_function, apply_on_output=True), normalizer,
        #     apply_on_output=True)
        # new_callback_generator = bimode_normalize(
        #     bimode_augment(callback_generator, noise_function=noise_function, apply_on_output=True),
        #     normalizer, apply_on_output=True)

        new_train_generator = bimode_normalize(train_generator, normalizer, apply_on_output=True)
        new_val_generator = bimode_normalize(val_generator, normalizer, apply_on_output=True)
        new_callback_generator = bimode_normalize(callback_generator, normalizer, apply_on_output=True)

        return new_train_generator, new_val_generator, new_callback_generator

    def get_categorizer_augmentation():
        new_train_generator = normalize_generator(
            augment_generator(train_generator, noise_function=noise_function, apply_on_output=False),
            normalizer, apply_on_output=False)

        # new_val_generator = normalize_generator(val_generator, normalizer, apply_on_output=False)
        new_val_generator = normalize_generator(
            augment_generator(val_generator, noise_function=noise_function, apply_on_output=False),
            normalizer, apply_on_output=False)

        new_callback_generator = normalize_generator(
            augment_generator(callback_generator, noise_function=noise_function, apply_on_output=False),
            normalizer, apply_on_output=False)

        # new_train_generator = rotsym_augmentor(new_train_generator)
        # new_val_generator = rotsym_augmentor(new_val_generator)
        # new_callback_generator = rotsym_augmentor(new_callback_generator)

        return new_train_generator, new_val_generator, new_callback_generator

    def get_regressor_augmentation():
        return train_generator, val_generator, callback_generator

    print('ADD DATA-AUGMENTATION')

    normalizer = NORMALIZATION_FUNCTIONS[config[task].get('NORMALIZATION', 'intensity')]()[0]
    noise_function = NOISE_FUNCTIONS['composition'](config[task].get('NOISE', None))

    functions = {"style_transfer": get_im2im_data_augmentation,
                 "saliency": get_im2im_data_augmentation,
                 "heatmap": get_im2im_data_augmentation,
                 "autoencoder": get_im2im_data_augmentation,
                 "flat_autoencoder": get_im2im_data_augmentation,
                 "categorizer": get_categorizer_augmentation,
                 "inceptionV3": get_categorizer_augmentation,
                 "bimode": get_bimode_augmentation,
                 "multiscale_bimode": get_bimode_augmentation,
                 "regressor": get_regressor_augmentation,
                 }
    return functions[task]()


def get_models(config, task, name, train_folder, default_input_shape=DEFAULT_INPUT_SHAPE, load=False):
    def get_saliency_model():
        if len(labels) == 2:
            config[task]['OUTPUT_CANALS'] = 1
        else:
            config[task]['OUTPUT_CANALS'] = len(labels)
        model = import_flat_model(name=name, config=config[task], load=load)
        model.callback_titles = ['Input', 'Prediction', 'Truth'] + labels
        model.class_weight = None
        return model

    def get_autoencoder_model():
        if task == 'flat_autoencoder':
            model = import_flat_model(name=name, config=config[task], load=load)
        else:
            model = import_unet_model(name=name, config=config[task], load=load)
        model.callback_titles = ['Input', 'Prediction', 'Truth'] + labels
        model.class_weight = None
        return model

    def get_categorizer_model():
        if task == 'inceptionV3':
            model = InceptionV3(input_shape, len(labels), name, load=load, imagenet=config[task].get('IMAGENET', False),
                                class_weight=class_weight)
        else:
            model = import_categorizer(len(labels), config=config[task], name=name, load=load,
                                       class_weight=class_weight)
        model.labels = labels
        return model

    def get_bimode_model():
        model = import_bimode(labels, config=config[task], name=name, load=load)
        model.labels = labels
        model.class_weight = None
        return model

    def get_multiscale_bimode_model():
        model = import_multiscale_bimode(labels, config=config[task], name=name, load=load)
        model.labels = labels
        model.class_weight = None
        return model

    def get_regressor_model():
        model = InceptionV3(input_shape, output_canals, name, load=load, imagenet=config[task].get('IMAGENET', False),
                            last_activation='linear', loss='mse', metrics=[])
        model.class_weight = None
        return model

    def get_heatmap_model():
        model = import_heatmap_model(name=name, config=config[task], load=load)
        model.callback_titles = ['Input', 'Prediction', 'Truth']
        model.class_weight = None
        return model

    labels = [folder for folder in os.listdir(train_folder) if os.path.isdir(os.path.join(train_folder, folder))]
    class_weight = [len(os.listdir(os.path.join(train_folder, folder))) for folder in labels]
    input_shape = config[task].get('INPUT_SHAPE', default_input_shape)
    output_canals = config[task].get('OUTPUT_CANALS')
    functions = {"saliency": get_saliency_model,
                 "autoencoder": get_autoencoder_model,
                 "heatmap": get_heatmap_model,
                 "flat_autoencoder": get_autoencoder_model,
                 "style_transfer": get_autoencoder_model,
                 "categorizer": get_categorizer_model,
                 "inceptionV3": get_categorizer_model,
                 "bimode": get_bimode_model,
                 "multiscale_bimode": get_multiscale_bimode_model,
                 "regressor": get_regressor_model,
                 }
    return functions[task]()


def get_callbacks(config, task, model, callback_generator):
    def get_im2im_callbacks():
        callbacks = [ModelCheckpoint(model.weight_filename, save_best_only=True),
                     HistoryCallback(),
                     AutoencoderExampleCallback(callback_generator, denormalizer=denormalizer)]
        return callbacks

    def get_bimode_callbacks():
        callbacks = [ModelCheckpoint(model.weight_filename, save_best_only=True),
                     BimodeHistoryCallback(),
                     BimodeExampleCallback(callback_generator, denormalizer=denormalizer),
                     ConfusionCallback(callback_generator, model.labels)]
        return callbacks

    def get_categorizer_callbacks():
        callbacks = [ModelCheckpoint(model.weight_filename, save_best_only=True),
                     HistoryCallback(),
                     ConfusionCallback(callback_generator, model.labels),
                     ClassificationExampleCallback(callback_generator, denormalizer=denormalizer)]
        return callbacks

    def get_regressor_callback():
        callbacks = [ModelCheckpoint(model.weight_filename, save_best_only=True),
                     HistoryCallback(),
                     GanRegressorExampleCallback(callback_generator,
                                                 gan_filename=config[task]["GAN_FILENAME"],
                                                 layer_number=config[task].get('LAYER_NUMBER', 1),
                                                 truncation_psi=config[task].get("TRUNCATION_PSI", 1),
                                                 denormalizer=denormalizer
                                                 )]
        return callbacks

    print(config[task])
    denormalizer = NORMALIZATION_FUNCTIONS[config[task].get('NORMALIZATION', 'intensity')]()[1]
    print(config[task].get('NORMALIZATION', 'intensity'))

    print('PUTING LANDMARKS')
    functions = {"saliency": get_im2im_callbacks,
                 "autoencoder": get_im2im_callbacks,
                 "flat_autoencoder": get_im2im_callbacks,
                 "style_transfer": get_im2im_callbacks,
                 "categorizer": get_categorizer_callbacks,
                 "inceptionV3": get_categorizer_callbacks,
                 "bimode": get_bimode_callbacks,
                 "multiscale_bimode": get_bimode_callbacks,
                 "regressor": get_regressor_callback,
                 }
    return functions[task]()


def main(task, dataset, batch_size=BATCH_SIZE, epochs=EPOCHS,
         training_steps=TRAINING_STEPS, validation_steps=VALIDATION_STEPS, initial_epoch=INITIAL_EPOCH,
         **kwargs):
    config = get_config()
    for key, value in kwargs.items():
        config[task][key] = value
    task = config[task]['TASK']

    train_folder, val_folder = get_dataset_roots(task, dataset=dataset)
    train_generator, val_generator, callback_generator, train_dir = get_generators(config, task, dataset, batch_size)
    train_generator, val_generator, callback_generator = get_data_augmentation(config, task, train_generator,
                                                                               val_generator, callback_generator)
    name = config[task].get('NAME', f'{dataset}_{task}')
    model = get_models(config, task, name, train_folder, load=initial_epoch != 0)
    callbacks = get_callbacks(config, task, model, callback_generator)
    print('FIRIN MAH LASER')
    train(model, train_generator, val_generator, callbacks,
          epochs=epochs, training_steps=training_steps, validation_steps=validation_steps, initial_epoch=initial_epoch)


def train(model, train_generator, val_generator, callbacks, training_steps=TRAINING_STEPS,
          validation_steps=VALIDATION_STEPS, epochs=EPOCHS, initial_epoch=INITIAL_EPOCH):
    model.fit_generator(generator=train_generator,
                        validation_data=val_generator,
                        verbose=1,
                        steps_per_epoch=training_steps,
                        validation_steps=validation_steps,
                        epochs=epochs,
                        callbacks=callbacks,
                        initial_epoch=initial_epoch,
                        # class_weight=model.class_weight
                        )


if __name__ == '__main__':
    fire.Fire(main)
