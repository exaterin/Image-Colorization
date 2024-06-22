from Edge_extraction import extract

import os
import numpy as np
from PIL import Image
from tqdm import tqdm

def process_images(input_folder, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for filename in tqdm(os.listdir(input_folder)):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(input_folder, filename)
            image = Image.open(image_path)
            sketch = extract.xdog(image)
            sketch_path = os.path.join(output_folder, filename)
            sketch.save(sketch_path)

# Specify the paths to your image folders
input_folder = 'images'
output_folder = 'sketches'

# Process the images
process_images(input_folder, output_folder)