import numpy as np
import matplotlib.pyplot as plt
from skimage.color import lab2rgb

def read_ab_pairs(filename):
    ab_pairs = []
    with open(filename, 'r') as file:
        for line in file:
            # Strip unwanted characters and split by comma
            clean_line = line.strip().replace('(', '').replace(')', '')
            a, b = map(float, clean_line.split(','))
            ab_pairs.append([a, b])
    return ab_pairs


def visualize_ab_pairs_scatter(ab_pairs, L=50):
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

    # Show the color plot
    plt.show()

def plot_image(image):
    plt.figure(figsize=(10, 10))
    plt.imshow(image)
    plt.axis('off')  # Hide axes
    plt.show()

# Example usage
filename = 'Dataset/unique_ab_pairs.txt'  # Update this to your actual filename
ab_pairs = read_ab_pairs(filename)
image = visualize_ab_pairs_scatter(ab_pairs)
# plot_image(image)
