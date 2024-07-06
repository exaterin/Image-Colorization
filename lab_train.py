import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import torch.nn.functional as F
from torch.utils.data import random_split, DataLoader
import logging
import os
from datetime import datetime

from Datasets.dataset_lab import DatasetLAB
from Models.lab_model import CNNModel


from Datasets.utils import get_img_from_one_hot, save_image

parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", default=32, type=int, help="Batch size.")
parser.add_argument("--epochs", default=10, type=int, help="Number of epochs.")
parser.add_argument("--lr", default=0.001, type=float, help="Learning rate.")

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

        for images, labels, _ in train_loader_tqdm:
            images, labels = images.to(device).float(), labels.to(device).float()
            images = images.permute(0, 3, 1, 2)
            labels = labels.argmax(dim=3)
            optimizer.zero_grad()
            outputs = model(images)
            
            # Resize labels to match output size
            labels = F.interpolate(labels.unsqueeze(1).float(), size=outputs.shape[2:], mode='nearest').long().squeeze(1)
            
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            train_loader_tqdm.set_postfix(loss=loss.item())

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, labels, _ in dev_loader:
                images, labels = images.to(device).float(), labels.to(device).float()
                images = images.permute(0, 3, 1, 2)
                labels = labels.argmax(dim=3)
                
                # Resize labels to match output size
                labels = F.interpolate(labels.unsqueeze(1).float(), size=outputs.shape[2:], mode='nearest').long().squeeze(1)
                
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

        logging.info(f"Model: {model_name}, Epoch {epoch+1}/{epochs}, Training Loss: {running_loss/len(train_loader)}, Validation Loss: {val_loss/len(dev_loader)} ")

        print(f"Epoch {epoch+1}/{epochs}, Training Loss: {running_loss/len(train_loader)}, Validation Loss: {val_loss/len(dev_loader)}")

    torch.save(model.state_dict(), model_name)

def test(model_path, test_loader, dataset):
    model = CNNModel()
    model.load_state_dict(torch.load(model_path))
    model.to(device)
    model.eval()

    with torch.no_grad():
        for img, label, image_name in tqdm(test_loader, desc="Testing"):
            img, label = img.to(device).float(), label.to(device).float()
            img = img.permute(0, 3, 1, 2)
            labels = label.argmax(dim=3)

            outputs = model(img)
            
            # Resize labels to match output size
            labels = F.interpolate(labels.unsqueeze(1).float(), size=outputs.shape[2:], mode='nearest').long().squeeze(1)
            
            for lab_pred, grey_scale_img, image_name in zip(outputs, img, image_name):
                lab_pred = lab_pred.permute(1, 2, 0)
                grey_scale_img = grey_scale_img.permute(1, 2, 0)

                image = get_img_from_one_hot(grey_scale_img.cpu(), lab_pred.cpu(), ab_classes=dataset.ab_classes)
                save_image(image, image_name, directory=f'Image-Colorisation/output_{model_path}')

def main(args):
    setup_logging('Image-Colorisation/logs')

    dataset = DatasetLAB('images', sketch=True, ab_classes_path='Image-Colorisation/Datasets/ab_classes.txt', sketches_folder='sketches')


    l, onehot, name = dataset[0]

    img = get_img_from_one_hot(l, onehot, dataset.ab_classes)
    save_image(img, name, directory=f'Image-Colorisation/test')


    # # Split the dataset into train, test and val sets
    # generator = torch.Generator().manual_seed(42)
    # train_dataset, test_dataset, val_dataset = random_split(dataset, [0.8, 0.1, 0.1], generator=generator)

    # train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, pin_memory=True, num_workers=4)
    # dev_loader = DataLoader(val_dataset, batch_size=args.batch_size)
    # test_loader = DataLoader(test_dataset, batch_size=args.batch_size)

    # model = CNNModel()
    # criterion = nn.CrossEntropyLoss()
    # optimizer = optim.Adam(model.parameters(), lr=args.lr)


    # model_name = 'Model_lab_sketches.pth'

    # logging.info(f"Model file: {model_name}")

    # # train(model, train_loader, dev_loader, crls
    # # iterion, optimizer, args.epochs, model_name)

    # test(model_path=model_name, test_loader=test_loader, dataset=dataset)

if __name__ == '__main__':
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)
