import numpy as np
from PIL import Image
from skimage import color
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


def save_image(image, image_name, directory):
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
    non_black_pixels = np.where(image_array >= black_value)
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
    

