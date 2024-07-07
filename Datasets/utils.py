import os
import torch
import numpy as np
from PIL import Image
from skimage import color
import cv2
from tqdm import tqdm
import torchvision.transforms as transforms
from torchvision.models import inception_v3

from Edge_extraction.extract import xdog

def resize_and_pad(image, desired_size=256):
    """
    Resize and pad an image to the given size.
    
    Parameters:
    image (PIL Image): Input image.
    desired_size (int): Desired size of the output image.
    
    Returns:
    PIL Image: Resized and padded image.
    """
    ratio = float(desired_size) / max(image.size)
    new_size = tuple([int(x * ratio) for x in image.size])
    image = image.resize(new_size, Image.LANCZOS)
    
    # Create a new image and paste the resized image on it
    new_image = Image.new("RGB", (desired_size, desired_size))
    new_image.paste(image, ((desired_size - new_size[0]) // 2, (desired_size - new_size[1]) // 2))
    
    return new_image


def create_sketches(input_folder, output_folder):
    """
    Create sketches from images using the XDoG and save them to the output folder.

    Parameters:
    input_folder (str): The folder containing the input images.
    output_folder (str): The folder to save sketch images.
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


def save_rgb_image(rgb_image, image_name, directory):
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
    
    rgb_image = rgb_image.permute(1, 2, 0).cpu().numpy()

    # Ensure the values are between 0 and 1
    rgb_image = np.clip(rgb_image, 0, 1)

    rgb_image = (rgb_image * 255).astype(np.uint8)  # Convert to uint8
    Image.fromarray(rgb_image, mode='RGB').save(file_path)


def apply_gaussian_blur(img, sigma=60):
    """
    Apply Gaussian blur to an image. Used for creating reference images.

    Parameters:
    img (PIL Image): Image to be blurred.
    sigma (int): Standard deviation for Gaussian kernel.
    
    Returns:
    numpy array: Blurred image.
    """
    img = np.array(img)

    # Apply Gaussian blur
    # Kernel size should be odd and based on sigma
    ksize = (sigma * 2 + 1, sigma * 2 + 1)
    blurred_image = cv2.GaussianBlur(img, ksize, sigma)

    return blurred_image


def extract_features(image):
    """
    Extracts feature vector for a single image using pre-trained Inception-ResNet-v2 model.

    Parameters:
    image: Input image.

    Returns:
    np.array: Feature vector of the input image.
    """
    def process_image(image):

        # Resize and pad the image
        image = resize_and_pad(image)

        image = image.convert('L')
        image_np = np.array(image, dtype=np.uint8)

        # Stack grayscale image 3 times
        image_np = np.stack((image_np,) * 3, axis=-1)

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

    image = process_image(image)

    with torch.no_grad():
        features = model(image).numpy()

    return features
