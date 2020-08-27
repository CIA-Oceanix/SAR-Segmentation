import numpy as np
import os
import glob

from Rignak_DeepLearning.data import read
from Rignak_Misc.path import list_dir, convert_link

BATCH_SIZE = 8
INPUT_SHAPE = (512, 512, 3)
OUTPUT_SHAPE = (32, 32, 1)
INPUT_LABEL = "input"
OUTPUT_LABEL = "output"

DUMMY_INITIAL_BATCH = 20000
DUMMY_PROBABILITY_INCREASE = 0  # 0.05 / 2000  # incease of 5% each 2000 batchs
DUMMY_MAXIMUM_PROBABILITY = 0  # 0.5

def get_heatmap_generator_with_dummy_data(dummy_probability_increase=DUMMY_PROBABILITY_INCREASE,
                                          dummy_initial_batch=DUMMY_INITIAL_BATCH,
                                          dummy_maximum_probability=DUMMY_MAXIMUM_PROBABILITY):
    dummy_initial_batch = dummy_initial_batch if dummy_initial_batch is not None else DUMMY_INITIAL_BATCH
    dummy_maximum_probability = dummy_maximum_probability \
        if dummy_maximum_probability is not None else DUMMY_MAXIMUM_PROBABILITY
    dummy_probability_increase = dummy_probability_increase if \
        dummy_probability_increase is not None else DUMMY_PROBABILITY_INCREASE

    print(f'Will add dummy data at batch {dummy_initial_batch}')

    function_variables = {"probability": 0, "batch": 0}

    def heatmap_generator(root, batch_size=BATCH_SIZE, input_shape=INPUT_SHAPE, output_shape=OUTPUT_SHAPE,
                          input_label=INPUT_LABEL, output_label=OUTPUT_LABEL):
        up_folder, generator_type = os.path.split(root)
        up_folder, label = os.path.split(os.path.split(root)[0])

        alternative_roots = [os.path.join(up_folder, folder) for folder in os.listdir(up_folder) if folder != label]

        input_filenames = np.array(sorted(glob.glob(os.path.join(root, input_label, '*.*'))))
        output_filenames = np.array(sorted(glob.glob(os.path.join(root, output_label, '*.*'))))
        for filename in input_filenames:
            other_filename = filename.replace('.png - Raccourci.lnk', '.npy').replace(input_label, output_label)
            if other_filename not in output_filenames:
                print(other_filename)
        for filename in output_filenames:
            other_filename = filename.replace('.npy', '.png - Raccourci.lnk').replace(output_label, input_label)
            if other_filename not in input_filenames:
                print(other_filename)
        print(input_label, len(input_filenames), len(output_filenames))
        [convert_link(filename) for filename in input_filenames if filename.endswith('.lnk')]
        [convert_link(filename) for filename in output_filenames if filename.endswith('.lnk')]

        dummy_input_filenames = np.array(sorted([filename for root in alternative_roots
                                                 for filename in glob.glob(os.path.join(root, generator_type,
                                                                                        input_label, '*.*'))]))

        assert len(input_filenames) == len(output_filenames)
        yield None
        while True:
            batch_index = np.random.randint(0, len(input_filenames), size=batch_size)
            batch_input_path = input_filenames[batch_index]

            if len(dummy_input_filenames):
                batch_dummy_index = np.random.randint(0, len(dummy_input_filenames), size=batch_size)
                batch_dummy_path = dummy_input_filenames[batch_dummy_index]
            else:
                batch_dummy_path = [''] * batch_size

            choose_dummy = np.random.random(batch_size) < function_variables['probability']
            batch_input_path = np.array([e2 if boolean else e1 for e1, e2, boolean in zip(batch_input_path,
                                                                                          batch_dummy_path,
                                                                                          choose_dummy)])
            batch_input = np.array([read(path, input_shape) for path in batch_input_path])

            batch_output_path = output_filenames[batch_index]
            batch_output = np.array([read(path, output_shape)
                                     if not choose_dummy[i] else np.zeros(output_shape, dtype='uint8')
                                     for i, path in enumerate(batch_output_path)])
            # print(choose_dummy, batch_output.min(), batch_output.max(), batch_input.min(), batch_input.max())

            function_variables['batch'] += 1
            if function_variables['batch'] >= dummy_initial_batch and \
                    function_variables['probability'] < dummy_maximum_probability:
                function_variables["probability"] += dummy_probability_increase

                if not function_variables['batch'] % 2500:
                    print('The probability of dummy data is', round(function_variables["probability"], 3))
            yield batch_input, batch_output

    return heatmap_generator
