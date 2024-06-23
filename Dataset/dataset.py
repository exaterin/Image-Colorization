import os
import pandas as pd
from PIL import Image
import numpy as np
from skimage import color
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
import uuid

import utils

class ImageDataset(Dataset):
    def __init__(self, image_folder, ab_classes_path, device='cuda'):
        self.image_folder = image_folder
        self.image_files = [f for f in os.listdir(image_folder) if os.path.isfile(os.path.join(image_folder, f))]
        self.ab_classes = utils.read_ab_pairs(ab_classes_path)

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_name = self.image_files[idx]
        image_path = os.path.join(self.image_folder, image_name)
        image = Image.open(image_path).convert('RGB')  # Ensure image is in RGB

        # Resize and pad the image
        image = utils.resize_and_pad(image)

        # Convert to Lab
        lab_image = utils.to_lab(image)
        
        # Quantize ab channels
        ab_channels = utils.quantize_ab_channels(lab_image)

        # Map ab channels to classes
        ab_classes = utils.map_ab_to_class(ab_channels, self.ab_classes)

        # One-hot encode ab classes
        one_hot_ab_classes = utils.one_hot_encode(ab_classes, len(self.ab_classes))

        # Get the L channel
        L_channel = lab_image[:, :, 0]
        # Add a channel dimension
        L_channel = L_channel[:, :, np.newaxis]

        return L_channel, one_hot_ab_classes
    

if __name__ == '__main__':
    dataset = ImageDataset('Dataset/Images', ab_classes_path='Dataset/ab_classes.txt')

    # Show the first image
    image = dataset[5]

    image = utils.get_img_from_one_hot(image[0], image[1], dataset.ab_classes)

    print(image.shape)

    utils.show_image(image)