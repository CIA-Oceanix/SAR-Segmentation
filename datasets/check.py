import glob
from tqdm import tqdm
import cv2
import os

for filename in glob.glob('./**/desktop.ini', recursive=True):
    print(filename)
    os.remove(filename)

for filename in tqdm(glob.glob('./**/*.png', recursive=True)):
    im = cv2.imread(filename)
    if im is None:
        print(filename)
        os.remove(filename)
