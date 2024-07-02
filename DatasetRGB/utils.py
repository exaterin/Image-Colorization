import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt



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
    plt.imshow(rgb_image)
    plt.show()

    rgb_image = np.clip(rgb_image, 0, 1)  # Ensure the values are between 0 and 1
    rgb_image = (rgb_image * 255).astype(np.uint8)  # Convert to uint8
    Image.fromarray(rgb_image, mode='RGB').save(file_path)