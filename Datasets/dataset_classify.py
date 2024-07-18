import os
from PIL import Image
import numpy as np
from torch.utils.data import Dataset
import torch

from Datasets.utils import resize_and_pad

# Implementation of a dataset class for the classification model
class ImageSketchDataset(Dataset):
    def __init__(self, photos_dir, sketches_dir):
        self.photos_files = [os.path.join(photos_dir, f) for f in os.listdir(photos_dir) if os.path.isfile(os.path.join(photos_dir, f))]
        self.sketches_files = [os.path.join(sketches_dir, f) for f in os.listdir(sketches_dir) if os.path.isfile(os.path.join(sketches_dir, f))]
        self.files = self.photos_files + self.sketches_files
        self.labels = [0] * len(self.photos_files) + [1] * len(self.sketches_files)  # 0 for photo, 1 for sketch

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        image_path = self.files[idx]
        image = Image.open(image_path).convert('RGB')
        
        label = self.labels[idx]

        image = resize_and_pad(image)

        # Convert the RGB image to grayscale
        image = image.convert('L')
        image_array = np.array(image)

        # Add channel dimension
        tensor = torch.from_numpy(image_array).unsqueeze(0).float()

        image = tensor / 255.0

        return image, label