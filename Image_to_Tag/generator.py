import numpy as np
import os
import glob
import json

from Rignak_DeepLearning.data import read
from Rignak_Misc.path import convert_link

BATCH_SIZE = 8
INPUT_SHAPE = (256, 256, 3)


def tagger_base_generator(root, batch_size=BATCH_SIZE, input_shape=INPUT_SHAPE):
    filenames = np.array(sorted(glob.glob(os.path.join(root, '*.*'))))
    filenames = np.array([convert_link(filename) if filename.endswith('.lnk') else filename for filename in filenames])

    with open(os.path.join(os.path.split(root)[0], 'output.json')) as json_file:
        data = json.load(json_file)

    all_tags = []
    for key, tags in data.items():
        for tag in tags:
            if tag not in all_tags:
                all_tags.append(tag)
    all_tags = sorted(all_tags)
    data = {key: [1 if tag in current_tags else 0
                  for tag in all_tags]
            for key, current_tags in data.items()
            }

    yield all_tags, None
    while True:
        filenames_index = np.random.randint(0, len(filenames), size=batch_size)

        batch_filenames = filenames[filenames_index]
        batch_input = np.array([read(filename, input_shape) for filename in batch_filenames])
        batch_output = np.array([data[os.path.split(filename)[-1]] for filename in batch_filenames])
        yield batch_input, batch_output
