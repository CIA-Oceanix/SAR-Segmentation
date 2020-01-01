import pickle
import functools
import dnnlib.tflib as tflib

from Rignak_Misc.path import get_local_file

tflib.init_tf()

DEFAULT_MODEL = get_local_file(__file__, 'karras2019stylegan-ffhq-1024x1024.pkl')


@functools.lru_cache(maxsize=2)
def load_model(model_filename=DEFAULT_MODEL):
    with open(model_filename, 'rb') as f:
        _, _, Gs = pickle.load(f)
    return Gs
