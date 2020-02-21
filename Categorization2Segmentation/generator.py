import numpy as np
import os
import glob

from Rignak_DeepLearning.data import read

BATCH_SIZE = 8
INPUT_SHAPE = (512, 512, 3)
OUTPUT_SHAPE = (32, 32, 1)
INPUT_LABEL = "input"
OUTPUT_LABEL = "output"


def heatmap_generator(root, batch_size=BATCH_SIZE, input_shape=INPUT_SHAPE, output_shape=OUTPUT_SHAPE,
                      input_label=INPUT_LABEL, output_label=OUTPUT_LABEL):
    input_filenames = glob.glob(os.path.join(root, input_label, '*.png'))
    output_filenames = glob.glob(os.path.join(root, output_label, '*.png'))
    assert len(input_filenames) == len(output_filenames)
    while True:
        batch_index = np.random.randint(0, len(input_filenames), size=batch_size)

        batch_input_path = np.random.choice(input_filenames[batch_index], size=batch_size)
        batch_input = np.array([read(path, input_shape) for path in batch_input_path])

        batch_output_path = np.random.choice(output_filenames[batch_index], size=batch_size)
        batch_output = np.array([read(path, output_shape) for path in batch_output_path])
        yield batch_input, batch_output
