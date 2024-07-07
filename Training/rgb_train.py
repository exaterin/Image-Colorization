import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from tqdm import tqdm
import logging
import os
from datetime import datetime
from PIL import Image
import numpy as np

from torch.utils.data import random_split, DataLoader
from Models.rgb_model import ModelRGB
from Datasets.dataset_rgb import ImageDatasetRGB
from Datasets.utils import save_rgb_image


parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", default=32, type=int, help="Batch size.")
parser.add_argument("--epochs", default=10, type=int, help="Number of epochs.")
parser.add_argument("--lr", default=0.001, type=float, help="Learning rate.")
parser.add_argument("--image_folder", default="images", type=str, help="Path to the image folder.")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def setup_logging(log_dir):
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = os.path.join(log_dir, f'training_{timestamp}.log')
    logging.basicConfig(filename=log_file, level=logging.INFO, 
                        format='%(asctime)s - %(levelname)s - %(message)s')
    logging.info(f"Logging setup complete. Log file: {log_file}")


def train(model, train_loader, dev_loader, criterion, optimizer, epochs, model_name):
    model.to(device)
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        train_loader_tqdm = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")

        for gray_images, rgb_images, _ in train_loader_tqdm:
            gray_images, rgb_images = gray_images.to(device).float(), rgb_images.to(device).float()
            gray_images = gray_images.permute(0, 3, 1, 2)
            rgb_images = rgb_images.permute(0, 3, 1, 2)
            
            optimizer.zero_grad()
            outputs = model(gray_images)
            
            loss = criterion(outputs, rgb_images)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            train_loader_tqdm.set_postfix(loss=loss.item())
            

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for gray_images, rgb_images, _ in dev_loader:
                gray_images, rgb_images = gray_images.to(device).float(), rgb_images.to(device).float()
                gray_images = gray_images.permute(0, 3, 1, 2)
                rgb_images = rgb_images.permute(0, 3, 1, 2)
                
                outputs = model(gray_images)
                loss = criterion(outputs, rgb_images)
                val_loss += loss.item()

        
        logging.info(f"Model: {model_name}, Epoch {epoch+1}/{epochs}, Training Loss: {running_loss/len(train_loader)}, Validation Loss: {val_loss/len(dev_loader)} ")

        print(f"Epoch {epoch+1}/{epochs}, Training Loss: {running_loss/len(train_loader)}, Validation Loss: {val_loss/len(dev_loader)}")

    torch.save(model.state_dict(), f'Image-Colorisation/{model_name}')

def test(model_path, test_loader):
    model = ModelRGB()
    model.load_state_dict(torch.load(model_path))
    model.to(device)
    model.eval()

    with torch.no_grad():
        for gray_images, rgb_images, image_name in tqdm(test_loader, desc="Testing"):
            gray_images, rgb_images = gray_images.to(device).float(), rgb_images.to(device).float()
            gray_images = gray_images.permute(0, 3, 1, 2)
            rgb_images = rgb_images.permute(0, 3, 1, 2)

            outputs = model(gray_images)

            # print("Output min:", outputs.min().item(), "Output max:", outputs.max().item())
            
            for i in range(outputs.size(0)):
                save_rgb_image(outputs[i], image_name[i], f'Image-Colorisation/output_{model_path}')

def main(args):

    setup_logging('Image-Colorisation/logs')

    dataset = ImageDatasetRGB(args.image_folder, 'sketches')

    # Split the dataset into train, test and val sets
    generator = torch.Generator().manual_seed(42)
    train_dataset, test_dataset, val_dataset = random_split(dataset, [0.8, 0.1, 0.1], generator=generator)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, pin_memory=True, num_workers=4)
    dev_loader = DataLoader(val_dataset, batch_size=args.batch_size)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size)

    model = ModelRGB()
    criterion = nn.MSELoss()  # Using MSE loss for image reconstruction
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    model_name = 'Model_RGB.pth'

    logging.info(f"Model file: {model_name}")

    # train(model, train_loader, dev_loader, criterion, optimizer, args.epochs, model_name)

    test(model_path=model_name, test_loader=test_loader)

if __name__ == '__main__':
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)

    # dataset = ImageDatasetRGB('images', 'sketches')

    # sketch, rgb, name = dataset[0]
    # sketch = np.squeeze(sketch, axis=2)

    # sketch = (sketch * 255).astype(np.uint8)

    # sketch_image = Image.fromarray(sketch)
    # sketch_image.save(f"Image-Colorisation/{name}_sketch.png")
