def intensity_normalization(f=1 / 1):
    def normalize(im):
        return im * f

    def denormalize(im):
        return im / f

    return normalize, denormalize
