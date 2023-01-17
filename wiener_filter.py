"""Wiener-Filter"""
import random

import matplotlib.pyplot as plt
import numpy as np
from numpy.fft import fft2, ifft2


def rgb_to_gray(img: np.ndarray) -> np.ndarray:
    """Convert color image into grayscaled version
    
    Args:
        img: an numpy.ndarray representation of the image
        
    Returns:
        a grayscaled representation of the same image
    """
    return np.dot(img[..., :3], [0.2989, 0.5870, 0.1140])


def motion_blur(size: tuple[int], T: float, delta: tuple[float]) -> np.ndarray:
    """Create motion blur effect on image
    
    The returned degradation function has to be applied to the target in the
    frequency domain.
    
    Args:
        size: a tuple of ints for the image's size
        T: value for velocity of the motion
        delta: tuple of floats for the direction of the movement
        
    Returns:
        a numpy.ndarray representing the degradation function then applied to
        an image
    """
    x, y = size
    H = np.zeros((x + 1, y + 1), dtype=np.complex128)
    for u1 in range(1, x + 1):
        for u2 in range(1, y + 1):
            d_combine = np.pi * (u1 * delta[0] + u2 * delta[1])
            H[u1, u2] = T * np.sinc(d_combine) * np.exp(-1j * d_combine)
    return H[1:, 1:]


def gaussian_noise(img: np.ndarray, sigma: int) -> np.ndarray:
    """Apply Gaussian noise on image
    
    Args:
        img: numpy.ndarray of the image (grayscaled version should be used)
        sigma: standard deviation of the Gaussian bell curve
        
    Returns:
        blurred image (as numpy.ndarray)
    """
    gaussian_shape = np.random.normal(0, sigma, np.shape(img))
    new_img = img + gaussian_shape
    new_img[new_img < 0] = 0
    new_img[new_img > 255] = 255
    return new_img


def salt_and_pepper_noise(img: np.ndarray, intensity: float) -> np.ndarray:
    """Apply salt & pepper noise on image
    
    Args:
        img: numpy.ndarray of the image (grayscaled version should be used)
        intensity: relative number of affected pixels (in per cent)
        
    Returns:
        degraded image (as numpy.ndarray)
    """
    new_img = np.copy(img)
    with np.nditer(new_img, op_flags=["readwrite"]) as row:
        for pixel in row:
            if random.random() < intensity:
                pixel[...] = 0 if random.random() < 0.5 else 255
    return new_img


def wiener_filter(img: np.ndarray, H: np.ndarray, K: float) -> np.ndarray:
    """Apple Wiener-filter on image
    
    Args:
        img: numpy.ndarray of the image (grayscaled version should be used)
        H: restoration function
        K: combined SNR- and PS value
        
    Returns:
        restored image (as numpy.ndarray)
    """
    G = fft2(np.copy(img))
    H = fft2(H, s=img.shape)
    H = np.conj(H) / (np.abs(H) ** 2 + K)
    return np.abs(ifft2(G * H))


def mean_square_error(benchmark: np.ndarray, estimation: np.ndarray) -> float:
    """Calculate mean square error
    
    The benchmark image has to be of the same size as the estimation.
    
    Args:
        benchmark: the original image
        estimation: the estimated image
        
    Returns:
        the squared difference of both images
    """
    sx, sy = benchmark.shape
    sum = 0
    for x in range(sx):
        for y in range(sy):
            sum += (benchmark[x][y] - estimation[x][y]) ** 2
    return sum / (sx * sy)


if __name__ == "__main__":
    # read and convert image
    img_grayscaled = rgb_to_gray(plt.imread("lena.jpg"))
    
    # create degradation function
    blur = motion_blur(img_grayscaled.shape, 1, (0, 0.045))
    
    # apply impulse noise to image
    img_sap = salt_and_pepper_noise(img_grayscaled, 0.01)
    
    # apply Gaussian noise to image
    img_noisy = gaussian_noise(img_sap, 12)
    
    # apply degradation function to blurred image
    img_blurred = np.abs(ifft2(blur * fft2(img_noisy)))
    
    # apply Wiener-Filter on degraded image
    img_filtered = wiener_filter(img_blurred, ifft2(blur), 0.004)
    
    # present results
    figures = [img_grayscaled, img_noisy, img_blurred, img_filtered]
    fig = plt.figure(figsize=(8, 8))
    fig.canvas.manager.set_window_title("Anwendung des Wiener-Filters")
    for i, img in enumerate(figures):
        fig.add_subplot(2, 2, i + 1)
        plt.imshow(img, cmap="gray")
        plt.axis("off")
        plt.subplots_adjust(left=0.05, bottom=0.05, wspace=0.1, hspace=0.1)
    plt.show()
