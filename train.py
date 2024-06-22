import argparse
import torch
import os
from torch.utils.data import random_split
from torch.utils.data import DataLoader
from keras.models import load_model

from ModelCNN.cnn_model import CNNModel
from Dataset.dataset import ImageDataset

os.environ.setdefault("KERAS_BACKEND", "torch")

parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", default=32, type=int, help="Batch size.")
parser.add_argument("--epochs", default=10, type=int, help="Number of epochs.")


def train(train_loader, dev_loader):
    model = CNNModel()
    model.summary()

    model = model.to('cuda')

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(train_loader, epochs=args.epochs, validation_data=dev_loader, verbose=1)
    
    # torch.save(model.state_dict(), 'Model1.pth')
    model.save('Model1.keras')


def test(model_path, test_loader, dataset):
    # # Load the model
    model = load_model(model_path, custom_objects={'CNNModel': CNNModel})
    model = model.to('cuda')

    for img, label in test_loader:
        img = img.to('cuda')
        output = model.predict(img)

        for lab_pred, grey_scale_img in zip(output, img):
            # print(lab.shape)
            # print(grey_scale_img.shape)
            print(label.shape)
            image = dataset.get_img_from_one_hot(grey_scale_img, lab_pred)

            dataset.show_image(image)



def main(args):
    dataset = ImageDataset('/Users/ekaterinalipina/images/1000', 'Dataset/ab_classes.txt')

    # Split the dataset into train, test and val sets
    generator = torch.Generator().manual_seed(42)
    train_dataset, test_dataset, val_dataset = random_split(dataset, [0.8, 0.1, 0.1], generator=generator)

    train_loader = torch.utils.data.DataLoader(train_dataset, args.batch_size, shuffle=True)
    dev_loader = torch.utils.data.DataLoader(val_dataset, args.batch_size)
    test_loader = torch.utils.data.DataLoader(test_dataset, args.batch_size)

    train(train_loader, dev_loader)

    test(model_path='Model1.keras', test_loader=test_loader, dataset=dataset)


    

if __name__ == '__main__':
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)