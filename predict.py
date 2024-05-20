from keras.models import load_model
from Dataset.dataset import ImageDataset
from ModelCNN.cnn_model import CNNModel
import numpy as np
import torch


dataset = ImageDataset('Dataset/Images', ab_classes_path='Dataset/ab_classes.txt')



# Load the model
model = load_model('Model1.keras', custom_objects={'CNNModel': CNNModel})

inp, _ = dataset[5]

if len(inp.shape) == 3:
    inp = np.expand_dims(inp, axis=0)


model.summary()

print(inp[0].shape)

img = model(inp[0], training=False)

# img_tensor = torch.from_numpy(img)
# img = torch.squeeze(img_tensor, dim=2)

print(img.shape)
# Show the image
image = dataset.get_img_from_one_hot(dataset[5][0], img)


dataset.show_image(image)



