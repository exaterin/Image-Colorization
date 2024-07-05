import os
from PIL import Image
import numpy as np
from torch.utils.data import Dataset

from Datasets.utils import resize_and_pad

class DatasetRefcCNN(Dataset):
    def __init__(self, image_folder, reference_folder, sketch_folder=None):
        self.image_folder = image_folder
        self.reference_folder = reference_folder
        self.image_files = [f for f in os.listdir(image_folder) if os.path.isfile(os.path.join(image_folder, f))]

        if sketch_folder:
            self.sketch_folder = sketch_folder

        self.reference_files = [f for f in os.listdir(reference_folder) if os.path.isfile(os.path.join(reference_folder, f))]

    def __len__(self):
        return len(self.image_files)

    # def __getitem__(self, idx):
    #     image_name = self.image_files[idx]
    #     image_path = os.path.join(self.image_folder, image_name)
    #     ref_image_path = os.path.join(self.reference_folder, self.reference_files[idx])
        
    #     image = Image.open(image_path).convert('RGB')  # Ensure image is in RGB
    #     ref_image = Image.open(ref_image_path).convert('RGB')  # Ensure reference image is in RGB

    #     # Resize and pad the image
    #     image = resize_and_pad(image)
    #     ref_image = resize_and_pad(ref_image)

    #     # Convert image to numpy array
    #     image_np = np.array(image)
    #     ref_image_np = np.array(ref_image)

    #     # Get grayscale image
    #     gray_image = Image.fromarray(image_np).convert('L')
    #     gray_image = np.array(gray_image).astype(np.float32) / 255.0
    #     gray_image = gray_image[:, :, np.newaxis]  # Add a channel dimension

    #     # Normalize RGB images
    #     rgb_image = image_np.astype(np.float32) / 255.0
    #     ref_image = ref_image_np.astype(np.float32) / 255.0

    #     img_name = os.path.basename(image_path).split('.')[0]

    #     return gray_image, rgb_image, ref_image, img_name



    def __getitem__(self, idx):
        image_name = self.image_files[idx]
        image_path = os.path.join(self.image_folder, image_name)
        ref_image_path = os.path.join(self.reference_folder, self.reference_files[idx])
        
        image = Image.open(image_path).convert('RGB')  # Ensure image is in RGB
        ref_image = Image.open(ref_image_path).convert('RGB')  # Ensure reference image is in RGB

        # Resize and pad the image
        image = resize_and_pad(image)
        ref_image = resize_and_pad(ref_image)

        # Convert image to numpy array
        image_np = np.array(image)
        ref_image_np = np.array(ref_image)


        sketch_path = os.path.join(self.sketch_folder, image_name)
        sketch = Image.open(sketch_path).convert('RGB')
        sketch = resize_and_pad(sketch)
        sketch_np = np.array(sketch)

        sketch_image = Image.fromarray(sketch_np).convert('L')
        sketch_image = np.array(sketch_image).astype(np.float32) / 255.0
        sketch_image = sketch_image[:, :, np.newaxis]  # Add a channel dimension


        # Normalize RGB images
        rgb_image = image_np.astype(np.float32) / 255.0
        ref_image = ref_image_np.astype(np.float32) / 255.0

        img_name = os.path.basename(image_path).split('.')[0]

        return sketch_image, rgb_image, ref_image, img_name