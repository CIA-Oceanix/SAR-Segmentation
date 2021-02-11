import numpy as np
import os
import glob
import json

from Rignak_DeepLearning.data import read
from Rignak_Misc.path import convert_link

BATCH_SIZE = 8
INPUT_SHAPE = (256, 256, 3)


def regressor_base_generator(root, attributes, batch_size=BATCH_SIZE, input_shape=INPUT_SHAPE):
    if isinstance(attributes, str):
        while ' ' in attributes:
            attributes = attributes.replace(' ', '')
        if attributes.startswith('(') and attributes.endswith(')'):
            attributes = attributes[1:-1]
        attributes = attributes.split(',')

    filenames = sorted(glob.glob(os.path.join(root, '*.png')))
    filenames += sorted(glob.glob(os.path.join(root, '*', '*.png')))
    filenames = np.array(filenames)
    
    
    with open(os.path.join(os.path.split(root)[0], 'output.json')) as json_file:
        data = json.load(json_file)

    checked_filenames = []
    for filename in filenames:
        if filename.endswith('.lnk'):
            filename = convert_link(filename)
        # if os.path.split(filename)[-1] not in data:
        #     print(filename, os.path.split(filename)[-1])
        if all([os.path.split(filename)[-1] in data
                and attribute in data[os.path.split(filename)[-1]]
                and not np.isnan(data[os.path.split(filename)[-1]][attribute])
                for attribute in attributes]):
            checked_filenames.append(filename)
    filenames = np.array(checked_filenames)
    print(attributes)
    
    with open(os.path.join(os.path.split(root)[0], 'normalization.json')) as json_file:
        normalization = json.load(json_file)
        means = [normalization[attribute]['mean'] for attribute in attributes]
        stds = [normalization[attribute]['std'] for attribute in attributes]
        
    print(f'The attributes were defined for {len(filenames)} files')
    print('MEANS:', means)
    print('STDS:', stds)
    yield tuple(means), tuple(stds)
    while True:
        filenames_index = np.random.randint(0, len(filenames), size=batch_size)

        batch_filenames = filenames[filenames_index]
        batch_input = np.array([read(filename, input_shape) for filename in batch_filenames])
        batch_output = np.array([[data[os.path.split(filename)[-1]][attribute]
                                  for attribute in attributes]
                                 for filename in batch_filenames])
        batch_output = (batch_output - means) / stds
        yield batch_input, batch_output

