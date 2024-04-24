import os
import pandas as pd
from PIL import Image
import numpy as np
from skimage import color
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset

class ImageDataset(Dataset):
    def __init__(self, image_folder):
        self.image_folder = image_folder
        self.image_files = [f for f in os.listdir(image_folder) if os.path.isfile(os.path.join(image_folder, f))]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_name = self.image_files[idx]
        image_path = os.path.join(self.image_folder, image_name)
        image = Image.open(image_path).convert('RGB')  # Ensure image is in RGB

        image = ImageDataset.resize_and_pad(image) # Resize and pad the image
        image = self.to_lab(image)  # Convert to Lab
        ab_channels = self.quantize_ab_channels(image) # Quantize ab channels
        L_channel = image[:, :, 0] # Get the L channel
        L_channel = L_channel[:, :, np.newaxis] # Add a channel dimension

        print(L_channel.shape, ab_channels.shape)

        return L_channel, ab_channels

    def to_lab(self, image):
        """
        Convert an RGB image to Lab color space.
        """
        image = np.array(image)
        lab_image = color.rgb2lab(image)
        return lab_image

    def quantize_ab_channels(self, lab_image):
        """
        Quantize ab channels of a Lab image.
        """
        ab_channels = lab_image[:, :, 1:]  # Extract ab channels
        ab_channels = (ab_channels // 10) * 10 + 5  # Quantization
        return ab_channels

    def get_unique_ab_pairs(self):
        """
        Get all unique (a, b) pairs across the dataset.
        """
        unique_ab_pairs = set()

        # 
        for idx in tqdm(range(len(self)), desc="Processing Images"):
            lab_image = self[idx]
            unique_pairs = set(map(tuple, lab_image.reshape(-1, 2)))  # Collect unique pairs from this image
            unique_ab_pairs.update(unique_pairs)

            print(len(unique_ab_pairs))

            if len(unique_ab_pairs) == 313:
                return unique_ab_pairs
            
        return unique_ab_pairs
    
    @staticmethod
    def split_dataset(image_folder, test_size=0.2, path='Code/Splits',random_state=None, save_csv=False, transform=None):
        '''
        Split the dataset into train and test sets
        '''
        image_files = [f for f in os.listdir(image_folder) if os.path.isfile(os.path.join(image_folder, f))]
        train_files, test_files = train_test_split(image_files, test_size=test_size, random_state=random_state)
        if save_csv:
            combined_list = [(file, 'train') for file in train_files] + [(file, 'test') for file in test_files]
            combined_df = pd.DataFrame(combined_list, columns=['filename', 'split'])
            if not os.path.exists(path):
                os.makedirs(path)
            combined_df.to_csv(f'{path}/combined_splits.csv', index=True)
        return ImageDataset(image_folder, train_files, transform), ImageDataset(image_folder, test_files, transform)

    @staticmethod
    def create_split_from_csv(image_folder, csv_file='Code/Splits/combined_splits.csv', transform=None):
        '''
        Create a split from a csv file.
        '''
        file_df = pd.read_csv(csv_file)
        train_files = file_df[file_df['split'] == 'train']['filename'].tolist()
        test_files = file_df[file_df['split'] == 'test']['filename'].tolist()
        return ImageDataset(image_folder, train_files, transform), ImageDataset(image_folder, test_files, transform)
    
    @staticmethod
    def read_ab_pairs(filename):
        ab_pairs = []
        with open(filename, 'r') as file:
            for line in file:
                clean_line = line.strip().replace('(', '').replace(')', '')
                a, b = map(float, clean_line.split(','))
                ab_pairs.append([a, b])
        return ab_pairs
    
    @staticmethod
    def resize_and_pad(image, desired_size=256):
        
        ratio = float(desired_size) / max(image.size)
        new_size = tuple([int(x * ratio) for x in image.size])
        image = image.resize(new_size, Image.LANCZOS)
        
        # Create a new image and paste the resized on it
        new_image = Image.new("RGB", (desired_size, desired_size))
        new_image.paste(image, ((desired_size - new_size[0]) // 2, (desired_size - new_size[1]) // 2))
        
        return new_image


def show_image(image):
    """
    Show an image.
    """
    lab_image = color.lab2rgb(image)
    Image.fromarray((lab_image * 255).astype(np.uint8)).show()


if __name__ == '__main__':
    dataset = ImageDataset('Dataset/Images')

    # Show the first image
    image = dataset[0]

    # show_image(image)