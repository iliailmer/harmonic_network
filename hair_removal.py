from skimage import color, morphology
from skimage.filters import threshold_otsu
import numpy as np
from .enhancement import rescale


def hair_removal(image, radius=3):
    luv = color.rgb2luv(image)
    L = luv[:, :, 0]
    # u = luv[:, :, 1]
    # v = luv[:, :, 2]
    morphed = np.zeros_like(luv)
    for i in range(3):
        morphed[:, :, i] = morphology.closing(
            luv[:, :, i], selem=morphology.disk(radius))

    diff = np.abs(L - morphed[:, :, 0])
    threshold = diff > threshold_otsu(diff)

    notted = np.zeros_like(luv)
    for i in range(3):
        notted[:, :, i] = luv[:, :, i]*(~threshold)

    multiplied = np.zeros_like(luv)
    for i in range(3):
        multiplied[:, :, i] = morphed[:, :, i]*threshold

    result = rescale(color.luv2rgb(multiplied+notted))
    return result
