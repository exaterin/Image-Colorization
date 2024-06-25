import os
import glob
import numpy as np
import argparse
import cv2 as cv
from skimage.io import imsave
from skimage.color import rgb2gray
from cv2 import GaussianBlur, imread
import matplotlib.pyplot as plt
import time

from PIL import Image
import numpy as np
import cv2 as cv
from skimage.color import rgb2gray


# Constants for XDoG line extraction
GAMMA = 0.95
SIGMA = 1
K = 3
EPSILON = -0.05
PHI = 100

# Constants for Sobel line extraction
THRESH = 100

# Constants for Canny line extraction
THRESH1 = 100
THRESH2 = 200

def xdog(image, gamma=GAMMA, sigma=SIGMA, k=K, epsilon=EPSILON, phi=PHI):
    """
    Apply XDoG edge detection to an image.
    
    Parameters:
    image (numpy array): Input image.
    gamma (float): Gamma value for XDoG.
    sigma (float): Sigma value for Gaussian blur.
    k (float): Multiplier for sigma for the second Gaussian blur.
    epsilon (float): Epsilon value for thresholding.
    phi (float): Phi value for tanh function.
    
    Returns:
    PIL Image: XDoG edge detected image.
    """

    image = np.array(image)
    if image.ndim == 3 and image.shape[2] == 3:
        image = rgb2gray(image)

    image1 = cv.GaussianBlur(image, (0, 0), sigma)
    image2 = cv.GaussianBlur(image, (0, 0), sigma * k)

    difference = image1 - gamma * image2
    mask = difference < epsilon

    difference[mask] = 1
    difference[~mask] = 1 + np.tanh(phi * difference[~mask])

    normalized_difference = (difference - difference.min()) / (difference.max() - difference.min())
    return Image.fromarray((normalized_difference * 255).astype(np.uint8))

def sobel(image, thresh=THRESH):
    """
    Apply Sobel edge detection to an image.
    
    Parameters:
    image (numpy array): Input image.
    thresh (int): Threshold value for edge detection.
    
    Returns:
    PIL Image: Sobel edge detected image.
    """
    img = np.array(image)
    if img.ndim == 3 and img.shape[2] == 3:
        img = cv.cvtColor(img, cv.COLOR_RGB2GRAY)

    mx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    gx = cv.filter2D(img, -1, mx)
    gy = cv.filter2D(img, -1, np.rot90(mx))
    edges = np.hypot(gx, gy) >= thresh

    return Image.fromarray((255 - (edges.astype(np.uint8) * 255)))

def canny(image, thresh1=THRESH1, thresh2=THRESH2):
    """
    Apply Canny edge detection to an image.
    
    Parameters:
    image (numpy array): Input image.
    thresh1 (int): First threshold for the hysteresis procedure.
    thresh2 (int): Second threshold for the hysteresis procedure.
    
    Returns:
    PIL Image: Canny edge detected image.
    """
    img = np.array(image)
    if img.ndim == 3 and img.shape[2] == 3:
        img = cv.cvtColor(img, cv.COLOR_RGB2GRAY)

    edges = cv.Canny(img, thresh1, thresh2)
    return Image.fromarray(255 - edges)
