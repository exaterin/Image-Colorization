import streamlit as st
from PIL import Image
import numpy as np
import sys
import os
import torch
from PIL import Image, ImageEnhance

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../Image Colorization')))

from Datasets.utils import resize_and_pad, apply_gaussian_blur, extract_features

from Models.rgb_model import ModelRGB
from Models.ccnn_model import RefcCNNModel
from Models.inception_model import ModelInception
from Models.classify_model import ImageClassifier


# Function to load a model based on its type
def load_model(model_type, image_type=None):

    if model_type == 'Classifier':
        model = ImageClassifier()
        model_path = "Weights/Model_Classification.pth"

    elif model_type == 'Base Model':
        model = ModelRGB()

        if image_type == 'photo':
            model_path = "Weights/Model_RGB.pth"

        else:
            model_path = "Weights/Model_RGB_sketch.pth"


    elif model_type == 'User-guided Model':
        model = RefcCNNModel()

        if image_type == 'photo':
            model_path = "Weights/Ref_Model_RGB.pth"
        else:
            model_path = "Weights/Ref_Model_RGB_sketch.pth"

    elif model_type == 'Model with Resnet-Inception features':
        model = ModelInception()

        if image_type == 'photo':
            model_path = "Weights/Model_Inception.pth"
        else:
            model_path = "Weights/Model_Inception_sketch.pth"

    else:
        raise ValueError("Unknown model type")

    # Load the model weights
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    return model

# Function to preprocess images before they are fed into the model
def preprocess_image(image, reference=False):
    image = resize_and_pad(image)
    if not reference:
        image = image.convert('L')
    
    image_np = np.array(image).astype(np.float32) / 255.0

    if not reference:
        image_np = image_np[np.newaxis, :, :]  # Add channel dimension for grayscale
    else:
        image_np = image_np.transpose((2, 0, 1))  # Change from HWC to CHW format for RGB

    image_np = image_np[np.newaxis, :, :, :]  # Add batch dimension

    return image_np

# Function to postprocess the output from the model
def postprocess_image(output, original_image):

    output = output.squeeze().permute(1, 2, 0).detach().numpy()
    output = (output * 255).astype(np.uint8)

    output = crop_to_original_size(output, original_image)

    pil = Image.fromarray(output)

    enhanced_image = enhance(pil)

    return enhanced_image

# Function to enhance the visual appeal of an image
def enhance(image):
    # Convert to NumPy array and subtract 50 from pixel values
    image_np = np.array(image).astype(np.float32)
    image_np = np.clip(image_np - 110, 0, 255).astype(np.uint8)
    
    # Convert back to PIL Image
    image = Image.fromarray(image_np)

    # Adjust brightness
    brightness_enhancer = ImageEnhance.Brightness(image)
    bright_image = brightness_enhancer.enhance(3)  # Increase brightness by a factor of 1.5

    saturation_enhancer = ImageEnhance.Color(bright_image)
    enhanced_image = saturation_enhancer.enhance(0.5) 

    return enhanced_image

# Function to crop an image to match the original size
def crop_to_original_size(image_array, original_image):

    original_image.thumbnail((256, 256), Image.ANTIALIAS)
    
    width, height = original_image.size
    start_x = (image_array.shape[1] - width) // 2
    start_y = (image_array.shape[0] - height) // 2
    cropped_image = image_array[start_y:start_y + height, start_x:start_x + width]
    return cropped_image

# Function to classify images using a preloaded model
def classify(image):

    model = load_model('Classifier')
    grey_image = preprocess_image(image)

    grey_image_tensor = torch.tensor(grey_image)

    with torch.no_grad():
        output = model(grey_image_tensor)

    label = torch.argmax(output).item()

    return label

# Main function
def colorisation_app(image):
    st.markdown("### Image Colorisation Models")
    
    st.image(image, caption='Original Image', use_column_width=True)

    color_model = st.selectbox(
        'Choose a colorisation model',
        ('Base Model', 'User-guided Model', 'Model with Resnet-Inception features')
    )

    if color_model == 'Base Model':
        label = classify(image)

        # For photo
        if label == 0:
            model = load_model(color_model, 'photo')

        # Sketch
        else:
            model = load_model(color_model, 'sketch')

        grey_image = preprocess_image(image)

        if st.button('Apply Colorisation'):

            grey_image = preprocess_image(image)

            grey_image_tensor = torch.tensor(grey_image)

            with torch.no_grad():
                output = model(grey_image_tensor)

            colorized_image = postprocess_image(output, image)

            grey_image = image.convert('L')

            col1, col2 = st.columns(2)
            col1.image(grey_image , caption="Input Image", use_column_width=True)
            col2.image(colorized_image, caption="Colorized Image", use_column_width=True)

    elif color_model == 'User-guided Model':

        reference_file = st.file_uploader("Upload a reference image", type=["jpg", "png", "jpeg"])
        if reference_file is not None:
            reference_image = Image.open(reference_file)

        label = classify(image)

        # For photo
        if label == 0:
            model = load_model(color_model, 'photo')

        # Sketch
        else:
            model = load_model(color_model, 'sketch')

        if st.button('Apply Colorisation') and reference_file is not None:

            reference = apply_gaussian_blur(reference_image)
            reference_image = Image.fromarray(reference)
            reference_image = preprocess_image(reference_image, reference=True)

            grey_image = preprocess_image(image)

            reference_tensor = torch.tensor(reference_image)
            grey_tensor = torch.tensor(grey_image)

            with torch.no_grad():
                output = model(grey_tensor, reference_tensor)


            colorized_image = postprocess_image(output, image)

            grey_image = image.convert('L')

            col1, col2, col3 = st.columns(3)
            col1.image(grey_image , caption="Input Image", use_column_width=True)
            col2.image(reference , caption="Reference Image", use_column_width=True)
            col3.image(colorized_image, caption="Colorized Image", use_column_width=True)

    elif color_model == 'Model with Resnet-Inception features':
        label = classify(image)

        # For photo
        if label == 0:
            model = load_model(color_model, 'photo')

        # Sketch
        else:
            model = load_model(color_model, 'sketch')

        features = extract_features(image)

        grey_image = preprocess_image(image)

        if st.button('Apply Colorisation'):

            grey_image = preprocess_image(image)

            grey_image_tensor = torch.tensor(grey_image)

            features_tensor = torch.tensor(features).float()

            with torch.no_grad():
                output = model(grey_image_tensor, features_tensor)

            colorized_image = postprocess_image(output, image)

            grey_image = image.convert('L')

            col1, col2 = st.columns(2)
            col1.image(grey_image , caption="Input Image", use_column_width=True)
            col2.image(colorized_image, caption="Colorized Image", use_column_width=True)

    

            

