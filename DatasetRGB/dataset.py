import os
from PIL import Image
import numpy as np
from torch.utils.data import Dataset

from utils import resize_and_pad

class ImageDatasetRGB(Dataset):
    def __init__(self, image_folder):
        self.image_folder = image_folder
        self.image_files = [f for f in os.listdir(image_folder) if os.path.isfile(os.path.join(image_folder, f))]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_name = self.image_files[idx]
        image_path = os.path.join(self.image_folder, image_name)
        image = Image.open(image_path).convert('RGB')  # Ensure image is in RGB

        # Resize and pad the image
        image = resize_and_pad(image)

        # Convert image to numpy array
        image_np = np.array(image)

        # Get grayscale image
        gray_image = Image.fromarray(image_np).convert('L')
        gray_image = np.array(gray_image).astype(np.float32) / 255.0
        gray_image = gray_image[:, :, np.newaxis]  # Add a channel dimension

        # Normalize RGB image
        rgb_image = image_np.astype(np.float32) / 255.0

        return gray_image, rgb_image
    

if __name__ == '__main__':
    dataset = ImageDatasetRGB('DatasetLAB/Images')

    # Show the first image
    grey, rgb = dataset[5]

    print(grey.shape, rgb.shape)