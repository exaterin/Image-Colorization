import numpy as np
from PIL import Image
from skimage import color
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
import os
import uuid
from tqdm import tqdm
import cv2
import torchvision.transforms as transforms
from torchvision.models import inception_v3
import torch

from Edge_extraction.extract import xdog

def show_image(image):
    """
    Show an image.
    """
    lab_image = color.lab2rgb(image)
    Image.fromarray((lab_image * 255).astype(np.uint8)).show()

def read_ab_pairs(filename):
    """
    Read ab pairs from a file.
    
    Parameters:
    filename (str): The path to the file containing ab pairs.
    
    Returns:
    list: List of ab pairs.
    """
    ab_pairs = []
    with open(filename, 'r') as file:
        for line in file:
            clean_line = line.strip().replace('(', '').replace(')', '')
            a, b = map(float, clean_line.split(','))
            ab_pairs.append([a, b])
    return ab_pairs

def quantize_ab_channels(lab_image):
    """
    Quantize ab channels of a Lab image.
    
    Parameters:
    lab_image (numpy array): Lab image.
    
    Returns:
    numpy array: Quantized ab channels.
    """
    ab_channels = lab_image[:, :, 1:]  # Extract ab channels
    ab_channels = (ab_channels // 10) * 10 + 5  # Quantization
    return ab_channels


def resize_and_pad(image, desired_size=256):
    """
    Resize and pad an image to the desired size.
    
    Parameters:
    image (PIL Image): Image to be resized and padded.
    desired_size (int): Desired size of the output image.
    
    Returns:
    PIL Image: Resized and padded image.
    """
    ratio = float(desired_size) / max(image.size)
    new_size = tuple([int(x * ratio) for x in image.size])
    image = image.resize(new_size, Image.LANCZOS)
    
    # Create a new image and paste the resized on it
    new_image = Image.new("RGB", (desired_size, desired_size))
    new_image.paste(image, ((desired_size - new_size[0]) // 2, (desired_size - new_size[1]) // 2))
    
    return new_image

def to_lab(image):
    """
    Convert an RGB image to Lab color space.
    
    Parameters:
    image (PIL Image): RGB image.
    
    Returns:
    numpy array: Image in Lab color space.
    """
    image = np.array(image)
    lab_image = color.rgb2lab(image)
    return lab_image

def one_hot_encode(image, num_classes):
    """
    One-hot encode an image.
    
    Parameters:
    image (numpy array): Image to be one-hot encoded.
    num_classes (int): Number of classes.
    
    Returns:
    numpy array: One-hot encoded image.
    """
    one_hot_array = np.zeros((image.shape[0], image.shape[1], num_classes), dtype=np.uint8)
    
    # Generate the indices for each axis
    x = np.arange(image.shape[0]).reshape(image.shape[0], 1, 1)
    y = np.arange(image.shape[1]).reshape(1, image.shape[1], 1)

    # Use the class indices in 'image' to define the index for the third dimension
    class_indices = image.reshape(image.shape[0], image.shape[1], 1)
    
    # Use advanced indexing to set '1' in the correct locations
    one_hot_array[x, y, class_indices] = 1

    return one_hot_array

def map_ab_to_class(ab_channels, ab_classes):
    """
    Map ab channels to classes.
    
    Parameters:
    ab_channels (numpy array): ab channels of the image.
    ab_classes (list): List of ab classes.
    
    Returns:
    numpy array: Image with mapped ab classes.
    """
    ab_channels = ab_channels.reshape(-1, 2)  # Flatten ab channels
    ab_class_indices = np.zeros(ab_channels.shape[0], dtype=int)  # Placeholder for classes

    for idx, ab in enumerate(ab_channels):
        # If index exists
        if ab.tolist() in ab_classes:
            ab_class_indices[idx] = ab_classes.index(ab.tolist())
        # Find the nearest class
        else:
            distances = np.linalg.norm(np.array(ab_classes) - ab, axis=1)
            ab_class_indices[idx] = np.argmin(distances)

    # Reshape back to original shape (256, 256, 1)
    ab_class_indices = ab_class_indices.reshape(256, 256, 1)

    return ab_class_indices

def get_img_from_one_hot(l_channel, one_hot_ab_classes, ab_classes):
    """
    Get an image from one-hot encoded ab classes.
    
    Parameters:
    l_channel (numpy array): L channel of the image.
    one_hot_ab_classes (numpy array): One-hot encoded ab classes.
    ab_classes (list): List of ab classes.
    
    Returns:
    numpy array: Reconstructed Lab image.
    """
    ab = np.argmax(one_hot_ab_classes, axis=2)
    ab_channels = np.array(ab_classes)[ab]

    image = np.concatenate((l_channel, ab_channels), axis=2)

    return image

def save_image(image, image_name, directory='Image-Colorisation/output'):
    """
    Save an image to a file.
    
    Parameters:
    image (numpy array): Image to be saved.
    directory (str): Directory to save the image.
    """
    if not os.path.exists(directory):
        os.makedirs(directory)

    file_path = os.path.join(directory, f"{image_name}.png")

    lab_image = color.lab2rgb(image)
    rgb_image = (lab_image * 255).astype(np.uint8)
    Image.fromarray(rgb_image).save(file_path)


def create_sketches(input_folder, output_folder):
    """
    Create sketches from images using the XDoG edge detection method and save them to the output folder.

    Parameters:
    input_folder (str): The folder containing the input images.
    output_folder (str): The folder where the generated sketches will be saved.
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for filename in tqdm(os.listdir(input_folder)):
        if filename.lower().endswith(('.jpg')):
            image_path = os.path.join(input_folder, filename)
            image = Image.open(image_path)
            sketch = xdog(image)
            sketch_path = os.path.join(output_folder, filename)
            sketch.save(sketch_path)


def save_rgb_image(rgb_image, image_name, directory='output_images'):
    """
    Save an RGB image to a file.
    
    Parameters:
    - rgb_image (Tensor): RGB image to be saved.
    - image_name (str): Name of the image file.
    - directory (str): Directory to save the image.
    """
    if not os.path.exists(directory):
        os.makedirs(directory)

    file_path = os.path.join(directory, f"{image_name}.png")
    
    rgb_image = rgb_image.permute(1, 2, 0).cpu().numpy()  # Convert to NumPy array

    rgb_image = 2 * (rgb_image - 0.3)

    rgb_image = np.clip(rgb_image, 0, 1)  # Ensure the values are between 0 and 1

    rgb_image = (rgb_image * 255).astype(np.uint8)  # Convert to uint8
    Image.fromarray(rgb_image, mode='RGB').save(file_path)


def crop_black_borders(image_array, black_value):

    # Find the bounding box of the non-black regions
    non_black_pixels = np.where(image_array > black_value)
    top = np.min(non_black_pixels[0])
    bottom = np.max(non_black_pixels[0])
    left = np.min(non_black_pixels[1])
    right = np.max(non_black_pixels[1])
    
    # Crop the image to the bounding box
    cropped_image = image_array[top:bottom+1, left:right+1]
    return cropped_image


def apply_gaussian_blur(input_folder, output_folder, sigma=60):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    files = [f for f in os.listdir(input_folder) if f.endswith(('.png', '.jpg', '.jpeg'))]
    
    for filename in tqdm(files, desc="Applying Gaussian Blur"):
        image_path = os.path.join(input_folder, filename)
        image = cv2.imread(image_path)

        # Apply Gaussian blur
        ksize = (sigma * 2 + 1, sigma * 2 + 1)  # Kernel size should be odd and based on sigma
        blurred_image = cv2.GaussianBlur(image, ksize, sigma)
        
        # Convert the result back to an image
        blurred_image = Image.fromarray(cv2.cvtColor(blurred_image, cv2.COLOR_BGR2RGB))
        output_path = os.path.join(output_folder, filename)
        blurred_image.save(output_path)


def extract_and_save_features(image_folder, output_folder):
    def process_image(image_path):
        image = Image.open(image_path).convert('RGB')  # Ensure image is in grayscale

        # Resize and pad the image
        image = resize_and_pad(image)

        # Convert the image to grayscale
        image = image.convert('L')

        # Convert to numpy array and stack to create a 3-channel image
        image_np = np.array(image, dtype=np.uint8)

        image_np = np.stack((image_np,)*3, axis=-1)  # Stack grayscale image 3 times

        # Convert numpy array to PIL image
        image_pil = Image.fromarray(image_np)

        # Transform the image to the required input size for Inception-ResNet-v2
        transform = transforms.Compose([
            transforms.Resize((299, 299)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        image = transform(image_pil)  # Add batch dimension

        return image.unsqueeze(0) 
    
    # Load the pre-trained Inception model
    model = inception_v3(pretrained=True)
    model.eval()

    # Remove the final classification layer
    model.fc = torch.nn.Identity()

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    image_files = [f for f in os.listdir(image_folder) if f.endswith(('.png', '.jpg', '.jpeg'))]

    for filename in tqdm(image_files, desc="Extracting features"):
        image_path = os.path.join(image_folder, filename)
        image = process_image(image_path)

        with torch.no_grad():
            features = model(image).numpy()

        feature_path = os.path.join(output_folder, filename.replace('.png', '.npy')
                                                  .replace('.jpg', '.npy')
                                                  .replace('.jpeg', '.npy'))
    
        np.save(feature_path, features)
    

