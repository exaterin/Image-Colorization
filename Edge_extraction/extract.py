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
    img = np.array(image)
    if img.ndim == 3 and img.shape[2] == 3:
        img = cv.cvtColor(img, cv.COLOR_RGB2GRAY)

    mx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    gx = cv.filter2D(img, -1, mx)
    gy = cv.filter2D(img, -1, np.rot90(mx))
    edges = np.hypot(gx, gy) >= thresh

    return Image.fromarray((255 - (edges.astype(np.uint8) * 255)))

def canny(image, thresh1=THRESH1, thresh2=THRESH2):
    img = np.array(image)
    if img.ndim == 3 and img.shape[2] == 3:
        img = cv.cvtColor(img, cv.COLOR_RGB2GRAY)

    edges = cv.Canny(img, thresh1, thresh2)
    return Image.fromarray(255 - edges)

# Plot the results
def show_plots(image_path):

    if image_path:
        # Load the first image
        original_image = imread(image_path)
        original_image = cv.cvtColor(original_image, cv.COLOR_BGR2RGB)

        # Apply edge extraction methods
        xdog_image = xdog(original_image)
        sobel_image = sobel(original_image)
        canny_image = canny(original_image)

    # Median of three edge detectors

        # Plotting the results
        fig, axes = plt.subplots(1, 4, figsize=(4, 4))
        axes[0].imshow(original_image)
        axes[0].set_title('Original Image')
        axes[0].axis('off')

        axes[1].imshow(xdog_image, cmap='gray')
        axes[1].set_title('XDoG Edge Detection')
        axes[1].axis('off')

        axes[2].imshow(sobel_image, cmap='gray')
        axes[2].set_title('Sobel Edge Detection')
        axes[2].axis('off')

        axes[3].imshow(canny_image, cmap='gray')
        axes[3].set_title('Canny Edge Detection')
        axes[3].axis('off')

        plt.show()

def show_edge_detection_grid(image_paths):
    fig, axes = plt.subplots(len(image_paths), 4, figsize=(12, 3 * len(image_paths)))
    
    for i, image_path in enumerate(image_paths):
        original_image = imread(image_path)
        if original_image.ndim == 3 and original_image.shape[2] == 3:
            original_image = cv.cvtColor(original_image, cv.COLOR_BGR2RGB)
        
        xdog_image = xdog(original_image)
        sobel_image = sobel(original_image)
        canny_image = canny(original_image)
        
        axes[i, 0].imshow(original_image)
        axes[i, 0].set_title('Original Image')
        axes[i, 0].axis('off')
        
        axes[i, 1].imshow(xdog_image, cmap='gray')
        axes[i, 1].set_title('XDoG Edge Detection')
        axes[i, 1].axis('off')
        
        axes[i, 2].imshow(sobel_image, cmap='gray')
        axes[i, 2].set_title('Sobel Edge Detection')
        axes[i, 2].axis('off')
        
        axes[i, 3].imshow(canny_image, cmap='gray')
        axes[i, 3].set_title('Canny Edge Detection')
        axes[i, 3].axis('off')
    
    plt.tight_layout()
    plt.show()

def show_time(image_paths):
    for i, image_path in enumerate(image_paths):
        if image_path:
            # Load the image
            original_image = imread(image_path)
            original_image = cv.cvtColor(original_image, cv.COLOR_BGR2RGB)

            # Apply edge extraction methods
            start_time = time.time()
            xdog_image = xdog(original_image)
            xdog_time = time.time() - start_time

            start_time = time.time()
            sobel_image = sobel(original_image)
            sobel_time = time.time() - start_time

            start_time = time.time()
            canny_image = canny(original_image)
            canny_time = time.time() - start_time

            # Print the times for each image
            print(f"Image {i + 1}:")
            print(f"XDoG time: {xdog_time:.4f} seconds")
            print(f"Sobel time: {sobel_time:.4f} seconds")
            print(f"Canny time: {canny_time:.4f} seconds")
            print()

def show_xdog_variations(image_path, params_list):
    original_image = imread(image_path)
    original_image = cv.cvtColor(original_image, cv.COLOR_BGR2RGB)

    fig, axes = plt.subplots(1, len(params_list) + 1, figsize=(12, 4))
    axes[0].imshow(original_image)
    axes[0].set_title('Original Image')
    axes[0].axis('off')

    for i, params in enumerate(params_list):
        gamma, sigma, k, epsilon, phi = params
        xdog_image = xdog(original_image, gamma=gamma, sigma=sigma, k=k, epsilon=epsilon, phi=phi)
        axes[i + 1].imshow(xdog_image, cmap='gray')
        axes[i + 1].set_title(f'({chr(97+i)})')
        axes[i + 1].axis('off')

    plt.show()


if __name__ == '__main__':

    # image_folder = 'Images'

    # IMAGE_INDEX = 34

    # image_files = glob.glob(os.path.join(image_folder, '*.jpg'))
    # image_path = image_files[IMAGE_INDEX] if image_files else None

    # show_plots(image_path)

    show_plots('plot/Q384032_wd3.jpg')

    # show_time(['plot/Q12512_wd0.jpg', 'plot/Q167193_wd0.jpg', 'plot/Q384032_wd3.jpg'])

    # params_list = [
    #     (0.95, 1, 3, -0.05, 100),
    #     (0.9, 1, 4, -0.6, 35),
    #     (0.85, 0.4, 7, -0.4, 70)
    # ]
    # show_xdog_variations('plot/Q12512_wd0.jpg')
