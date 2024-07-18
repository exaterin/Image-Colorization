import sys
import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import logging
from datetime import datetime
from torch.utils.data import DataLoader, random_split

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../Image Colorization')))

from Datasets.dataset_classify import ImageSketchDataset
from Models.classify_model import ImageClassifier

# Set up command line arguments
parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", default=32, type=int, help="Batch size for training.")
parser.add_argument("--epochs", default=5, type=int, help="Number of training epochs.")
parser.add_argument("--lr", default=0.001, type=float, help="Learning rate for optimizer.")
parser.add_argument("--photos_dir", default="images", type=str, help="Directory containing photo images.")
parser.add_argument("--sketches_dir", default="sketches", type=str, help="Directory containing sketch images.")
parser.add_argument("--log_dir", default="logs", type=str, help="Directory to save logs.")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Logging
def setup_logging(log_dir):
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = os.path.join(log_dir, f'training_{timestamp}.log')
    logging.basicConfig(filename=log_file, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logging.info("Logging setup complete. Log file: " + log_file)
    return log_file

# Training part
def train(model, train_loader, dev_loader, criterion, optimizer, epochs, log_file):
    model.to(device)
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # Calculate accuracy
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            running_loss += loss.item()

        train_accuracy = 100 * correct / total

        # Validation step
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in dev_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                val_loss += criterion(outputs, labels).item()

                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        val_accuracy = 100 * correct / total

        # Log training and validation loss and accuracy
        training_loss = running_loss / len(train_loader)
        validation_loss = val_loss / len(dev_loader)
        logging.info(f"Epoch {epoch+1}/{epochs}, Training Loss: {training_loss}, Training Accuracy: {train_accuracy}%, Validation Loss: {validation_loss}, Validation Accuracy: {val_accuracy}%")
        print(f"Epoch {epoch+1}/{epochs}, Training Loss: {training_loss}, Training Accuracy: {train_accuracy}%, Validation Loss: {validation_loss}, Validation Accuracy: {val_accuracy}%")

    # Save the trained model
    model_path = os.path.join("Model_Classification.pth")
    torch.save(model.state_dict(), model_path)
    logging.info("Model saved to " + model_path)

def main(args):
    log_file = setup_logging(args.log_dir)
    dataset = ImageSketchDataset(args.photos_dir, args.sketches_dir)

    # Split dataset into training and validation
    generator = torch.Generator().manual_seed(42)
    train_dataset, test_dataset, val_dataset = random_split(dataset, [0.8, 0.1, 0.1], generator=generator)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, pin_memory=True, num_workers=4)
    dev_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    model = ImageClassifier()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    train(model, train_loader, dev_loader, criterion, optimizer, args.epochs, log_file)

if __name__ == '__main__':
    args = parser.parse_args()
    main(args)