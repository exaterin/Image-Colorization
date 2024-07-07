import os
from PIL import Image
import numpy as np
from torch.utils.data import Dataset

from Datasets.utils import resize_and_pad

class ImageDatasetInception(Dataset):
    def __init__(self, image_folder, feature_folder,sketch=False, sketch_folder=None):
        self.image_folder = image_folder
        self.feature_folder = feature_folder
        self.image_files = [f for f in os.listdir(image_folder) if os.path.isfile(os.path.join(image_folder, f))]
        self.sketch = sketch
        self.sketch_folder = sketch_folder if sketch else None

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_name = self.image_files[idx]
        image_path = os.path.join(self.image_folder, image_name)
        feature_path = os.path.join(self.feature_folder, image_name.replace('.jpg', '.npy'))
        
        image = Image.open(image_path)

        # Resize and pad the image
        image = resize_and_pad(image)

        image_np = np.array(image)

        if self.sketch:
            sketch_path = os.path.join(self.sketch_folder, image_name)
            grey = Image.open(sketch_path)
        else:
            grey = image

        # Resize and pad the sketch or photo
        grey = resize_and_pad(grey)
        grey_np = np.array(grey)

        # Convert sketch or photo to greyscale and normalize
        grey_image = Image.fromarray(grey_np).convert('L')
        grey_image = np.array(grey_image).astype(np.float32) / 255.0
        grey_image = grey_image[:, :, np.newaxis]  # Add a channel dimension

        # Normalize RGB image
        rgb_image = image_np.astype(np.float32) / 255.0

        # Load feature vector
        feature_vector = np.load(feature_path).astype(np.float32)

        # Extract image name without extension
        img_name = os.path.basename(image_path).split('.')[0]

        return grey_image, rgb_image, feature_vector, img_name