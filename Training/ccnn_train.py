import os
import sys
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import logging
from datetime import datetime

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../Image Colorization')))

from torch.utils.data import random_split, DataLoader
from Models.ccnn_model import RefcCNNModel
from Datasets.dataset_ccnn import DatasetRefcCNN
from Datasets.utils import save_rgb_image

parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", default=32, type=int, help="Batch size.")
parser.add_argument("--epochs", default=10, type=int, help="Number of epochs.")
parser.add_argument("--lr", default=0.001, type=float, help="Learning rate.")
parser.add_argument("--image_folder", default="images", type=str, help="Path to the image folder.")
parser.add_argument("--sketch_folder", default="sketches", type=str, help="Path to the sketches folder.")
parser.add_argument("--reference_folder", default="references", type=str, help="Path to the reference image folder.")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Logging
def setup_logging(log_dir):
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = os.path.join(log_dir, f'training_{timestamp}.log')
    logging.basicConfig(filename=log_file, level=logging.INFO, 
                        format='%(asctime)s - %(levelname)s - %(message)s')
    logging.info(f"Logging setup complete. Log file: {log_file}")

# Training part
def train(model, train_loader, dev_loader, criterion, optimizer, epochs, model_name):
    model.to(device)
    for epoch in range(epochs):
        # Set the model to training mode
        model.train()
        running_loss = 0.0
        train_loader_tqdm = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")

        for grey_images, rgb_images, ref_images, _ in train_loader_tqdm:
            grey_images, rgb_images, ref_images = grey_images.to(device).float(), rgb_images.to(device).float(), ref_images.to(device).float()
            grey_images = grey_images.permute(0, 3, 1, 2)
            rgb_images = rgb_images.permute(0, 3, 1, 2)
            ref_images = ref_images.permute(0, 3, 1, 2)
            
            # Zero the gradients
            optimizer.zero_grad()
            outputs = model(grey_images, ref_images)  # Forward pass
            
            loss = criterion(outputs, rgb_images)
            loss.backward()
            optimizer.step() # Update weights
            running_loss += loss.item()
            train_loader_tqdm.set_postfix(loss=loss.item()) # Display loss
            
        # Set the model to evaluation mode
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for grey_images, rgb_images, ref_images, _ in dev_loader:
                grey_images, rgb_images, ref_images = grey_images.to(device).float(), rgb_images.to(device).float(), ref_images.to(device).float()
                grey_images = grey_images.permute(0, 3, 1, 2)
                rgb_images = rgb_images.permute(0, 3, 1, 2)
                ref_images = ref_images.permute(0, 3, 1, 2)
                
                outputs = model(grey_images, ref_images)
                loss = criterion(outputs, rgb_images)
                val_loss += loss.item()

        # Log the training and validation loss
        logging.info(f"Model: {model_name}, Epoch {epoch+1}/{epochs}, Training Loss: {running_loss/len(train_loader)}, Validation Loss: {val_loss/len(dev_loader)}")
        print(f"Epoch {epoch+1}/{epochs}, Training Loss: {running_loss/len(train_loader)}, Validation Loss: {val_loss/len(dev_loader)}")

    # Save the model
    torch.save(model.state_dict(), f'Weights/{model_name}')

# Testing part
def test(model_path, test_loader, output_images_path):
    model = RefcCNNModel()
    model.load_state_dict(torch.load(model_path))
    model.to(device)
    model.eval() # Set the model to evaluation mode

    with torch.no_grad():
        for grey_images, rgb_images, ref_images, image_name in tqdm(test_loader, desc="Testing"):
            grey_images, rgb_images, ref_images = grey_images.to(device).float(), rgb_images.to(device).float(), ref_images.to(device).float()
            grey_images = grey_images.permute(0, 3, 1, 2)
            rgb_images = rgb_images.permute(0, 3, 1, 2)
            ref_images = ref_images.permute(0, 3, 1, 2)

            outputs = model(grey_images, ref_images)

            for i in range(outputs.size(0)):
                save_rgb_image(outputs[i], image_name[i], output_images_path)

def main(args):

    setup_logging('logs')

    dataset = DatasetRefcCNN(args.image_folder, args.reference_folder, args.sketch_folder)

    # Split the dataset into train, test and val sets
    generator = torch.Generator().manual_seed(42)
    train_dataset, test_dataset, val_dataset = random_split(dataset, [0.8, 0.1, 0.1], generator=generator)

    # Create Data Loaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, pin_memory=True, num_workers=4)
    dev_loader = DataLoader(val_dataset, batch_size=args.batch_size)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size)

    model = RefcCNNModel()
    criterion = nn.MSELoss()  # Usingls MSE loss for image reconstruction
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    model_name = 'Ref_Model_RGB_sketch.pth'

    logging.info(f"Model file: {model_name}")

    train(model, train_loader, dev_loader, criterion, optimizer, args.epochs, model_name)

    test(model_path=model_name, test_loader=test_loader, output_images_path=f'output_{model_name}')

if __name__ == '__main__':
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)
