import os
import numpy as np
from PIL import Image
from tqdm import tqdm
from glob import glob

from Rignak_DeepLearning.OCR.apply_text_segmentation import OUTPUT_FOLDER
from Rignak_DeepLearning.generator import categorizer_generator

INPUT_SHAPE = (32, 32, 3)
INPUT_FOLDER = ""
CONV_LAYERS = (4, 8, 16, 32)

def resize(thumbnail, height=INPUT_SHAPE[0]):
    thumbnail_array = np.array(thumbnail)
    ratio = height / thumbnail_array.shape[0]
    width = int(ratio * thumbnail_array.shape[1])
    new_thumbnail = thumbnail.resize((width, height), Image.ANTIALIAS)
    return new_thumbnail

def create_dataset():
    pass

def main():
    mapping = {1: 'a', 2: 'b', 3: 'c', 4: 'd', 5: 'e', 6: 'f', 7: 'g', 8: 'h',
               9: 'i', 10: 'j', 11: 'k', 12: 'l', 13: 'm', 14: 'n', 15: 'o', 16: 'p',
               17: 'q', 18: 'r', 19: 's', 20: 't', 21: 'u', 22: 'v', 23: 'w', 24: 'x',
               25: 'y', 26: 'z'}
    inverse_mapping = {value: key for (key, value) in mapping.items()}




if __name__ == '__main__':
    main()
