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

from extract import xdog, sobel, canny

def show_plots(image_path):
    """
    Plot and display the original and edge detected images using XDoG, Sobel, and Canny methods.
    
    Parameters:
    image_path (str): Path to the input image.
    """

    if image_path:
        original_image = imread(image_path)
        original_image = cv.cvtColor(original_image, cv.COLOR_BGR2RGB)

        xdog_image = xdog(original_image)
        sobel_image = sobel(original_image)
        canny_image = canny(original_image)

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
    """
    Plot and display edge detected images for multiple input images.
    
    Parameters:
    image_paths (list): List of paths to input images.
    """

    for image_path in image_paths:
        original_image = imread(image_path)
        if original_image.ndim == 3 and original_image.shape[2] == 3:
            original_image = cv.cvtColor(original_image, cv.COLOR_BGR2RGB)
        
        xdog_image = xdog(original_image)
        sobel_image = sobel(original_image)
        canny_image = canny(original_image)
        
        fig, axes = plt.subplots(2, 2, figsize=(8, 8))
        
        axes[0, 0].imshow(original_image)
        axes[0, 0].set_title('Original Image')
        axes[0, 0].axis('off')
        
        axes[0, 1].imshow(xdog_image, cmap='gray')
        axes[0, 1].set_title('XDoG Edge Detection')
        axes[0, 1].axis('off')
        
        axes[1, 0].imshow(sobel_image, cmap='gray')
        axes[1, 0].set_title('Sobel Edge Detection')
        axes[1, 0].axis('off')
        
        axes[1, 1].imshow(canny_image, cmap='gray')
        axes[1, 1].set_title('Canny Edge Detection')
        axes[1, 1].axis('off')
    
        plt.tight_layout()
        plt.show()


def show_xdog_variations(image_path, params_list):
    """
    Display variations of XDoG edge detection with different parameters.
    
    Parameters:
    image_path (str): Path to the input image.
    params_list (list): List of tuples with XDoG parameters.
    """
    original_image = imread(image_path)
    original_image = cv.cvtColor(original_image, cv.COLOR_BGR2RGB)

    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    
    # Display the original image in the top left subplot
    axes[0, 0].imshow(original_image)
    axes[0, 0].set_title('Original Image')
    axes[0, 0].axis('off')

    for i, params in enumerate(params_list):
        gamma, sigma, k, epsilon, phi = params
        xdog_image = xdog(original_image, gamma=gamma, sigma=sigma, k=k, epsilon=epsilon, phi=phi)
        
        row, col = divmod(i + 1, 2)
        axes[row, col].imshow(xdog_image, cmap='gray')
        axes[row, col].set_title(f'({chr(97+i)})')
        axes[row, col].axis('off')

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':

    # Example parameters to try
    params_list = [
        (0.95, 1, 3, -0.05, 100),
        (0.9, 1, 4, -0.6, 35),
        (0.85, 0.4, 7, -0.4, 70)
    ]

    # Example usage
    show_xdog_variations('/Users/ekaterinalipina/images/images/Q282554_wd0.jpg', params_list)