from skimage.exposure import rescale_intensity
import numpy as np
from scipy import fftpack
from scipy.ndimage import maximum_filter, minimum_filter
from torch import Tensor, uint8


def rescale(image: np.ndarray, mn: int = 0, mx: int = 1):
    return rescale_intensity(image, out_range=(mn, mx))


def rescale_torch(image: Tensor) -> Tensor:
    return (255*(image-image.min())/(image.max()-image.min())).type(uint8)


def rot90(matrix: Tensor) -> Tensor:
    dims = range(len(matrix.shape))
    return matrix.transpose(dims[-2], dims[-1]).flip(2)


def weight_rotate(weight: Tensor, rot: int = 0) -> Tensor:
    if rot % 4 == 0:
        return weight
    if rot % 4 == 1:
        return rot90(weight)
    if rot % 4 == 2:
        return rot90(rot90(weight))
    if rot % 4 == 3:
        return rot90(rot90(rot90(weight)))


def edges(image: np.ndarray,
          size: int = 3) -> np.ndarray:
    image = rescale(image)
    ratio = np.zeros_like(image)
    if len(image.shape) > 2:
        ratio = np.divide(maximum_filter(image, (size, size, size))+1,
                          minimum_filter(image, (size, size, size))+1)
    else:
        ratio = np.divide(maximum_filter(image, (size, size))+1,
                          minimum_filter(image, (size, size))+1)
    ratio = rescale(20*np.log(ratio))
    return ratio


def add_edges(image: np.ndarray, a: float = 0.5, b: float = 0.5,
              size: int = 3) -> np.ndarray:
    ratio = np.zeros_like(image)
    if len(image.shape) > 2:
        ratio = np.divide(maximum_filter(image, (size, size, size))+1,
                          minimum_filter(image, (size, size, size))+1)
    else:
        ratio = np.divide(maximum_filter(image, (size, size))+1,
                          minimum_filter(image, (size, size))+1)
    ratio = rescale(20*np.log(ratio))
    image = rescale(image)
    return (a*image+b*ratio)/(a+b)


def EME(image):
    return np.sum(edges(image))/image.size


def alpha_rooting_fourier(image: np.ndarray, alpha: float = 0.9) -> np.array:
    ffted = fftpack.fft2(image)
    abs_ffted = np.absolute(ffted)**alpha
    iffted = fftpack.ifft2(abs_ffted*np.divide(ffted, np.absolute(ffted),
                                               out=np.zeros_like(ffted),
                                               where=np.absolute(ffted) != 0))
    iffted = rescale(np.absolute(iffted), 0, 1)  # .astype(int)
    return iffted
