import torch
from torch.utils.data import random_split
from torch.utils.data import DataLoader

from Dataset.dataset import ImageDataset
from ModelCNN.cnn_model import CNNModel


def main():
    dataset = ImageDataset('Dataset/Images')
    dataLoader = DataLoader(dataset=dataset, batch_size=32, shuffle=True)

    # Split the dataset into train, test and val sets
    generator = torch.Generator().manual_seed(42)
    train_dataset, test_dataset, val_dataset = random_split(dataset, [0.8, 0.1, 0.1], generator=generator)


    model = CNNModel()
    model.summary()

    # model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # model.fit(train_dataset, batch_size=32, epochs=10, validation_data=val_dataset)

    # model.evaluate(test_dataset)

    # model.save('ModelCNN/colornet.h5')


if __name__ == '__main__':
    main()