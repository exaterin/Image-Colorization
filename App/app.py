import streamlit as st
from PIL import Image
import io

from edge_extraction_app import edge_extraction_app
from colorization_app import colorisation_app

st.title('Image Processing App')

# Add a selectbox in the sidebar or at the top of the page
option = st.selectbox(
    'Choose an option',
    ('Edge Extraction Models', 'Image Colorisation Models')
)

uploaded_file = st.file_uploader("Choose a file", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    
    if option == 'Edge Extraction Models':
        edge_extraction_app(image)

    elif option == 'Image Colorisation Models':
        colorisation_app(image)
