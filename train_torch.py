import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import random_split, DataLoader
from tqdm import tqdm
from ModelCNN.cnn_model_torch import CNNModel
from DatasetLAB.dataset import ImageDataset
import torch.nn.functional as F

parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", default=32, type=int, help="Batch size.")
parser.add_argument("--epochs", default=10, type=int, help="Number of epochs.")
parser.add_argument("--lr", default=0.001, type=float, help="Learning rate.")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train(model, train_loader, dev_loader, criterion, optimizer, epochs):
    model.to(device)
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        train_loader_tqdm = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")

        for images, labels in train_loader_tqdm:
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
            for images, labels in dev_loader:
                images, labels = images.to(device).float(), labels.to(device).float()
                images = images.permute(0, 3, 1, 2)
                labels = labels.argmax(dim=3)
                
                # Resize labels to match output size
                labels = F.interpolate(labels.unsqueeze(1).float(), size=outputs.shape[2:], mode='nearest').long().squeeze(1)
                
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

        print(f"Epoch {epoch+1}/{epochs}, Training Loss: {running_loss/len(train_loader)}, Validation Loss: {val_loss/len(dev_loader)}")

    torch.save(model.state_dict(), 'Model_sketch.pth')

def test(model_path, test_loader, dataset):
    model = CNNModel()
    model.load_state_dict(torch.load(model_path))
    model.to(device)
    model.eval()

    with torch.no_grad():
        for img, label in tqdm(test_loader, desc="Testing"):
            img, label = img.to(device).float(), label.to(device).float()
            img = img.permute(0, 3, 1, 2)
            labels = label.argmax(dim=3)

            outputs = model(img)
            
            # Resize labels to match output size
            labels = F.interpolate(labels.unsqueeze(1).float(), size=outputs.shape[2:], mode='nearest').long().squeeze(1)
            
            for lab_pred, grey_scale_img in zip(outputs, img):
                lab_pred = lab_pred.permute(1, 2, 0)
                grey_scale_img = grey_scale_img.permute(1, 2, 0)

                image = dataset.get_img_from_one_hot(grey_scale_img.cpu(), lab_pred.cpu())
                dataset.save_image(image)

def main(args):
    dataset = ImageDataset('images', 'Image-Colorisation/Dataset/ab_classes.txt')

    # Split the dataset into train, test and val sets
    generator = torch.Generator().manual_seed(42)
    train_dataset, test_dataset, val_dataset = random_split(dataset, [0.8, 0.1, 0.1], generator=generator)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, pin_memory=True, num_workers=4)
    dev_loader = DataLoader(val_dataset, batch_size=args.batch_size)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size)

    model = CNNModel()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # train(model, train_loader, dev_loader, criterion, optimizer, args.epochs)

    test(model_path='Model1.pth', test_loader=test_loader, dataset=dataset)

if __name__ == '__main__':
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)
