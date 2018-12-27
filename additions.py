from skimage.exposure import rescale_intensity
import numpy as np
from scipy import fftpack
from scipy import absolute


def rescale(image: np.ndarray, mn: int = 0, mx: int = 1):
    return rescale_intensity(image, out_range=(mn, mx))


def EME(image: np.ndarray,
        window_height: int = 11,
        window_width: int = 11) -> float:
    """
    EME measure for showing image quality based on human visual system.
    For details, see
    Agaian, Sos S., Karen Panetta,
    and Artyom M. Grigoryan.
    "A new measure of image enhancement."
    IASTED International Conference on Signal Processing & Communication.
    Citeseer, 2000.

    :param image: input image, must be single-channel.
    :param window_height: height of the inspecting window
    :param window_width: width of the inspecting window
    :return: A real-valued enhancement measure.
    """
    height, width = image.shape
    sum_ = 0
    k = 0
    # range in height, distance from the center of the window
    H = np.int(np.floor(window_height / 2))
    W = np.int(np.floor(window_width / 2))  # range in width, same as above
    for row in range(0 + H, height - H + 1, window_height):
        for column in range(0 + W, width - W + 1, window_width):

            window = image[row - H:row + H + 1, column - W:column + W + 1]

            I_max = window.max()
            I_min = window.min()

            D = (I_max + 1) / (I_min + 1)
            if D < 0.02:
                D = 0.02
            k += 1
            sum_ += 20 * np.log(D)
        # sum_k_1 += sum_k_2
        # sum_k_2 = 0

    eme = sum_ / k
    return eme


def EME_color(image: np.ndarray) -> float:
    """
    Application of EME to color images.
    :param image: color image (multi-channel)
    :return: EME of that image
    """
    emes_ = []
    if image.shape[-1] > 1:
        for each in range(image.shape[-1]):
            emes_.append(EME(image[:, :, each]))
    else:
        emes_.append(EME(image))
    return max(emes_)


def alpha_rooting_fourier(image: np.ndarray, alpha: float = 0.9) -> np.ndarray:
    ffted = fftpack.fft2(image)
    abs_ffted = absolute(ffted)**alpha
    iffted = fftpack.ifft2(abs_ffted*ffted/absolute(ffted))
    iffted = rescale(absolute(iffted), 0, 1)  # .astype(int)
    return iffted
