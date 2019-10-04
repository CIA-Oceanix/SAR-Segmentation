import numpy as np
from PIL import Image

im = Image.open('reals.png')
imarray = np.array(im)

res = imarray.shape[1] // 15
while res > 4:
    new_im = im.copy()
    new_im.thumbnail((res*8, res*15))
    new_im.save(f'reals{res}.png')
    res /= 2
