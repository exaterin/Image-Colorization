import numpy as np
import matplotlib.pyplot as plt
from skimage.color import lab2rgb
from tqdm import tqdm

from dataset import ImageDataset


def get_unique_ab_pairs(dataset):
    """
    Get all unique (a, b) pairs across the dataset.

    Parameters:
    dataset (ImageDataset): The dataset containing images.

    Returns:
    list: Sorted list of unique (a, b) pairs.
    """
    unique_ab_pairs = set()

    for image in tqdm(dataset, desc="Processing Images"):
        lab_image, _ = image

        # Reshape the image to a 2D array and collect unique pairs
        unique_pairs = set(map(tuple, lab_image.reshape(-1, 2)))
        unique_ab_pairs.update(unique_pairs)

        # Check if all 313 pairs have been found
        if len(unique_ab_pairs) == 313:
            break

    # Sort ab pairs based on their Euclidean distance from the origin
    ab_values_sorted = sorted(unique_ab_pairs, key=lambda x: np.sqrt(x[0]**2 + x[1]**2))

    return ab_values_sorted


def save_ab_pairs(ab_pairs, filename):
    """
    Save the sorted (a, b) pairs to a file.

    Parameters:
    ab_pairs (list): List of sorted (a, b) pairs.
    filename (str): Filename to save the pairs.
    """

    with open(filename, 'w') as file:
        for ab in ab_pairs:
            file.write(f'({ab[0]}, {ab[1]})\n')

def visualize_ab_pairs_scatter(ab_pairs, L=50):
    """
    Visualize (a, b) pairs using a scatter plot.

    Parameters:
    ab_pairs (list): List of (a, b) pairs.
    L (int): Lightness value for converting to RGB.
    """

    # Extract separate lists for a and b values
    a_values, b_values = zip(*ab_pairs)

    # Create a scatter plot
    plt.figure(figsize=(8, 8))
    plt.scatter(b_values, a_values, c=[lab2rgb([[L, a, b]]) for a, b in ab_pairs], s=500, edgecolors='none', marker='s')

    # Set labels and title
    plt.xlabel('b')
    plt.ylabel('a')
    plt.title(f'RGB(a,b) | L = {L}')
    plt.gca().invert_yaxis()
    plt.grid(True)
    plt.axhline(0, color='black', linewidth=0.5)
    plt.axvline(0, color='black', linewidth=0.5)

    # Save as pdf
    plt.savefig('Dataset/ab_pairs.pdf')
    plt.show()


def create_ab_classes(dataset, filename):
    """
    Generate, save, and visualize unique (a, b) pairs from the dataset.

    Parameters:
    dataset (ImageDataset): The dataset containing images.
    filename (str): Filename to save the pairs.
    """

    # Get unique (a, b) pairs
    ab_pairs = get_unique_ab_pairs(dataset)

    # Save (a, b) pairs to a file
    save_ab_pairs(ab_pairs)

     # Visualize (a, b) pairs
    visualize_ab_pairs_scatter(ab_pairs, filename)


if __name__ == "__main__":
    dataset = ImageDataset('Dataset/Images', ab_classes_path=None)

    create_ab_classes(dataset, filename='Dataset/ab_classes.txt')