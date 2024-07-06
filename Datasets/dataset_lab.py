import os
from re import S
from PIL import Image
import numpy as np
from torch.utils.data import Dataset

from Datasets.utils import read_ab_pairs, resize_and_pad, to_lab, quantize_ab_channels, map_ab_to_class, one_hot_encode

class DatasetLAB(Dataset):
    def __init__(self, image_folder, ab_classes_path, sketch=False, sketches_folder=None):

        if sketch:
            self.sketch = True
        self.image_folder = image_folder
        self.image_files = [f for f in os.listdir(image_folder) if os.path.isfile(os.path.join(image_folder, f))]


        if sketches_folder and sketch:
            self.sketches_folder = sketches_folder

        if ab_classes_path:
            self.ab_classes = read_ab_pairs(ab_classes_path)

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_name = self.image_files[idx]
        image_path = os.path.join(self.image_folder, image_name)
        image = Image.open(image_path).convert('RGB')  # Ensure image is in RGB

        # Resize and pad the image
        image = resize_and_pad(image)

        # Convert to Lab
        lab_image = to_lab(image)
        
        # Quantize ab channels
        ab_channels = quantize_ab_channels(lab_image)

        # Map ab channels to classes
        ab_classes = map_ab_to_class(ab_channels, self.ab_classes)

        # One-hot encode ab classes
        one_hot_ab_classes = one_hot_encode(ab_classes, len(self.ab_classes))


        if self.sketch:
            sketch_path = os.path.join(self.sketches_folder, image_name)
            sketch = Image.open(sketch_path).convert('RGB')  # Ensure image is in RGB
            sketch = resize_and_pad(sketch)

            # Convert to Lab
            lab_sketch = to_lab(sketch)

            # Get the L channel
            L_channel = lab_sketch[:, :, 0]

        else:
            # Get the L channel
            L_channel = lab_image[:, :, 0]

        # Add a channel dimension
        L_channel = L_channel[:, :, np.newaxis]

        img_name = os.path.basename(image_path).split('.')[0]

        return L_channel, one_hot_ab_classes, img_name
    


    # def __getitem__(self, idx):
    #     image_name = self.image_files[idx]
    #     image_path = os.path.join(self.image_folder, image_name)
    #     image = Image.open(image_path).convert('RGB')  # Ensure image is in RGB

    #     # Resize and pad the image
    #     image = resize_and_pad(image)

    #     # Convert to Lab
    #     lab_image = to_lab(image)
        
    #     # Quantize ab channels
    #     ab_channels = quantize_ab_channels(lab_image)

    #     # Map ab channels to classes
    #     ab_classes = map_ab_to_class(ab_channels, self.ab_classes)

    #     # One-hot encode ab classes
    #     one_hot_ab_classes = one_hot_encode(ab_classes, len(self.ab_classes))

    #     img_name = os.path.basename(image_path).split('.')[0]

    #     return L_channel, one_hot_ab_classes, img_name
