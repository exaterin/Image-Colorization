from cgitb import grey
from json import load
import streamlit as st
from PIL import Image
import numpy as np
import sys
import os
import torch
from PIL import Image, ImageEnhance

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../Image Colorisation')))

from Datasets.utils import resize_and_pad

from Models.rgb_model import ModelRGB
from Models.ccnn_model import RefcCNNModel
from Models.inception_model import ModelInception
from Models.classify_model import ImageClassifier



def load_model(model_type, image_type=None):

    if model_type == 'Classifier':
        model = ImageClassifier()
        model_path = "Weights/Model_Classification.pth"

    elif model_type == 'Base Model':
        model = ModelRGB()
        model_path = "Weights/Model_RGB.pth"

    elif model_type == 'User-guided Model':
        model = RefcCNNModel()
        model_path = "Weights/Model_RGB.pth"

    elif model_type == 'Model with Resnet-Inception features':
        model = ModelInception()
        model_path = "Weights/Model_RGB.pth" 

    else:
        raise ValueError("Unknown model type")

    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    return model


def preprocess_image(image):
    image = resize_and_pad(image)
    image_np = np.array(image)
    grey_image = Image.fromarray(image_np).convert('L')
    grey_image = np.array(grey_image).astype(np.float32) / 255.0
    grey_image = grey_image[np.newaxis, :, :]  # Add channel dimension
    grey_image = grey_image[np.newaxis, :, :, :]  # Add batch dimension

    return grey_image

def postprocess_image(output, original_image):

    output = output.squeeze().permute(1, 2, 0).detach().numpy()
    output = (output * 255).astype(np.uint8)

    output = crop_to_original_size(output, original_image)

    pil = Image.fromarray(output)

    enhanced_image = enhance(pil)

    return enhanced_image


def enhance(image):
    enhancer = ImageEnhance.Contrast(image)
    enhanced_image = enhancer.enhance(2)

    image_np = np.array(enhanced_image)
    
    image_np = np.clip(image_np - 50, 0, 255).astype(np.uint8)
    enhanced_image = Image.fromarray(image_np)
    
    return enhanced_image


def crop_to_original_size(image_array, original_image):

    original_image.thumbnail((256, 256), Image.ANTIALIAS)
    
    width, height = original_image.size
    start_x = (image_array.shape[1] - width) // 2
    start_y = (image_array.shape[0] - height) // 2
    cropped_image = image_array[start_y:start_y + height, start_x:start_x + width]
    return cropped_image


def colorisation_app(image):
    st.markdown("### Image Colorisation Models")
    
    st.image(image, caption='Original Image', use_column_width=True)

    color_model = st.selectbox(
        'Choose a colorisation model',
        ('Base Model', 'User-guided Model', 'Model with Resnet-Inception features')
    )

    if color_model == 'Base Model':
        model = load_model('Classifier')
        grey_image = preprocess_image(image)

        grey_image_tensor = torch.tensor(grey_image)

        with torch.no_grad():
            output = model(grey_image_tensor)

        print(output)

        # model = load_model('Base Model')
        # grey_image = preprocess_image(image)

        # if st.button('Apply Colorisation'):
        #     if color_model == 'Base Model':

        #         grey_image = preprocess_image(image)

        #         grey_image_tensor = torch.tensor(grey_image)

        #         with torch.no_grad():
        #             output = model(grey_image_tensor)

        #         colorized_image = postprocess_image(output, image)

        #         grey_image = image.convert('L')

        #         col1, col2 = st.columns(2)
        #         col1.image(grey_image , caption="Input Image", use_column_width=True)
        #         col2.image(colorized_image, caption="Colorized Image", use_column_width=True)

    elif color_model == 'User-guided Model':

        reference_file = st.file_uploader("Upload a reference image", type=["jpg", "png", "jpeg"])
        if reference_file is not None:
            reference_image = Image.open(reference_file)
            st.image(reference_image, caption='Reference Image', use_column_width=True)

        model = load_model('User-guided Model')
        grey_image = preprocess_image(image)
