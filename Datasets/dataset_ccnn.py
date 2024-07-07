import os
from PIL import Image
import numpy as np
from torch.utils.data import Dataset

from Datasets.utils import resize_and_pad

class DatasetRefcCNN(Dataset):
    def __init__(self, image_folder, reference_folder, sketch=False, sketch_folder=None):
        self.image_folder = image_folder
        self.reference_folder = reference_folder
        self.image_files = [f for f in os.listdir(image_folder) if os.path.isfile(os.path.join(image_folder, f))]
        self.sketch = sketch
        self.sketch_folder = sketch_folder if sketch_folder else None
        self.reference_files = [f for f in os.listdir(reference_folder) if os.path.isfile(os.path.join(reference_folder, f))]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_name = self.image_files[idx]
        image_path = os.path.join(self.image_folder, image_name)
        ref_image_path = os.path.join(self.reference_folder, self.reference_files[idx])
        
        image = Image.open(image_path).convert('RGB')
        ref_image = Image.open(ref_image_path).convert('RGB')

        # Resize and pad images
        image = resize_and_pad(image)
        ref_image = resize_and_pad(ref_image)

        image_np = np.array(image)
        ref_image_np = np.array(ref_image)

        if self.sketch:
            sketch_path = os.path.join(self.sketch_folder, image_name)
            grey = Image.open(sketch_path)
        else:
            grey = image

        grey = resize_and_pad(grey)
        grey_np = np.array(grey)

        # Convert sketch or photo to greyscale and normalize
        grey_image = Image.fromarray(grey_np).convert('L')
        grey_image = np.array(grey_image).astype(np.float32) / 255.0
        grey_image = grey_image[:, :, np.newaxis]

        # Normalize RGB images
        rgb_image = image_np.astype(np.float32) / 255.0
        ref_image = ref_image_np.astype(np.float32) / 255.0

        # Extract image name without extension
        img_name = os.path.basename(image_path).split('.')[0]

        return grey_image, rgb_image, ref_image, img_name