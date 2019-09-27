import sys
import os
import numpy as np
from PIL import Image
import sounddevice
from scipy.io import wavfile

from Rignak_Misc.path import get_local_file
from Rignak_Misc.audio import short_time_fourier_transform, logscale_spectrogram

DATASET_ROOT = get_local_file(__file__, 'datasets')
WORD_FILENAME = 'words.txt'

SAMPLE_RATE = 44100
LENGTH = 3
CHANNELS = 2
SAMPLE_NUMBER = 10
FACTOR = 1
BIN_SIZE = 2 ** 10
SHAPE = (128, 128)


def record(sample_rate=SAMPLE_RATE, length=LENGTH, channels=CHANNELS):
    total_sample = int(length * sample_rate)
    recording = sounddevice.rec(total_sample, samplerate=sample_rate, channels=channels)
    print(f'Begin the recording for {length} seconds')
    print('End the recording\n')
    return recording


def write(folder, word, recording, sample, sample_rate=SAMPLE_RATE):
    full_filename = os.path.join(folder, word, f'{sample}.wav')
    wavfile.write(full_filename, sample_rate, recording)
    return full_filename


def parse_input(argvs):
    dataset = argvs[1]
    shape = SHAPE
    sample_number = SAMPLE_NUMBER
    sample_rate = SAMPLE_RATE
    dataset_root = DATASET_ROOT
    word_filename = WORD_FILENAME
    for argv in argvs[2:]:
        if argv.startswith('shape='):
            shape = int(argv.replace('shape=', ''))
            shape = (shape, shape)
        elif argv.startswith('sample_number='):
            sample_number = int(argv.replace('sample_number=', ''))
        elif argv.startswith('sample_rate='):
            sample_rate = int(argv.replace('sample_rate=', ''))
        elif argv.startswith('dataset_root='):
            dataset_root = int(argv.replace('dataset_root=', ''))
        elif argv.startswith('word_filename='):
            word_filename = argv.replace('word_filename=', '')
    return dataset, shape, sample_number, sample_rate, dataset_root, word_filename


def wav2png(wav, sample_rate=SAMPLE_RATE, bin_size=BIN_SIZE, factor=FACTOR):
    signal = short_time_fourier_transform(wav, bin_size)
    spectrogram, frequencies = logscale_spectrogram(signal, factor=factor, sample_rate=sample_rate)

    image = 20 * np.log10(np.abs(spectrogram) / 10 ** -6)  # conversion to decibels
    new_image = np.zeros((image.shape[0], image.shape[1], 3))
    new_image[:, :, :] = np.expand_dims(image, axis=-1)[:, :]
    return new_image.astype('uint8')


def main(dataset, shape, sample_number, sample_rate, dataset_root, word_filename):
    with open(os.path.join(dataset_root, dataset, word_filename), 'r') as file:
        words = [word[:-1] for word in file.readlines()]

    for word in words:
        print(f'Next word in {word}')
        folder = os.path.join(dataset_root, dataset, word)
        os.makedirs(folder, exist_ok=True)
        for i in range(sample_number):
            recording = record(sample_rate=sample_rate)
            filename = write(folder, word, recording, i, sample_rate=sample_rate)

            sample_rate, samples = wavfile.read(filename)
            image_array = wav2png(samples, sample_rate=sample_rate)
            image = Image.fromarray(image_array).resize(shape)
            image.save(filename.replace('.wav', '.png'))


if __name__ == '__main__':
    main(**parse_input(sys.argv))
