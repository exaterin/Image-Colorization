import os
import pandas as pd
from sklearn.model_selection import train_test_split
from torchvision import datasets
from torch.utils.data import Dataset
from PIL import Image

class ImageDataset(Dataset):
    def __init__(self, image_folder, image_files=None, transform=None):
        self.image_folder = image_folder
        self.transform = transform
        if image_files is None:
            self.image_files = [f for f in os.listdir(image_folder) if os.path.isfile(os.path.join(image_folder, f))]
        else:
            self.image_files = image_files

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_folder, self.image_files[idx])
        image = Image.open(img_path)  # Load the image
        image = image.convert('RGB')  # Convert to RGB format, if needed

        return image
    
    # Split the dataset into train and test sets
    @staticmethod
    def split_dataset(image_folder, test_size=0.2, random_state=None, save_csv=False, transform=None):
        image_files = [f for f in os.listdir(image_folder) if os.path.isfile(os.path.join(image_folder, f))]
        train_files, test_files = train_test_split(image_files, test_size=test_size, random_state=random_state)
        if save_csv:
            combined_list = [(file, 'train') for file in train_files] + [(file, 'test') for file in test_files]
            combined_df = pd.DataFrame(combined_list, columns=['filename', 'split'])
            combined_df.to_csv('combined_splits.csv', index=False)
        return ImageDataset(image_folder, train_files, transform), ImageDataset(image_folder, test_files, transform)

    # Create a split from a csv file
    @staticmethod
    def create_split_from_csv(image_folder, csv_file='combined_splits.csv', transform=None):
        file_df = pd.read_csv(csv_file)
        train_files = file_df[file_df['split'] == 'train']['filename'].tolist()
        test_files = file_df[file_df['split'] == 'test']['filename'].tolist()
        return ImageDataset(image_folder, train_files, transform), ImageDataset(image_folder, test_files, transform)

train_dataset, test_dataset = ImageDataset.split_dataset('Images', save_csv=True)