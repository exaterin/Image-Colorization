# Image Colorization Project

This repository contains the implementation of various image colorization models including a Base CNN model, a User-guided Model with reference images, and a CNN model enhanced with Inception-ResNet-v3 features. Additionally, the repository includes a model designed to classify images as either photos or sketches. A Streamlit application is also included to allow users to interactively try out and evaluate the implemented colorization methods.

## Project Overview

The goal of this project is to develop advanced models capable of colorizing both photos and hand-drawn sketches effectively. This repository not only explores various machine learning techniques and architectures for image colorization but also provides a Streamlit-based application for users to interactively experiment with these models.

## Features

- **Base CNN Model**: Implements a U-Net architecture for colorizing grayscale images.
- **User-guided Model**: Enhances the colorization process by using reference images to guide the colorization.
- **CNN with Inception Features**: Integrates high-level features from the Inception-ResNet-v3 model to improve colorization.
- **Image Classifier**: Automatically distinguishes between photos and sketches to apply the appropriate colorization model.
- **Interactive Application**: A Streamlit application that allows users to upload images, apply colorization and edge extraction models, and compare results.

## Installation

```bash
git clone https://github.com/yourusername/image-colorization.git
cd image-colorization
pip install -r requirements.txt

```


## Usage
To run a Streamlit application. 

```bash
streamlit run App/app.py
```

## Directory Structure

- **Models/**: Contains the implementations of the colorization and classification models.
- **Datasets/**: Includes scripts for loading and preprocessing the data.
- **Edge_extraction/**: Scripts for applying edge extraction methods to generate sketch images.
- **Weights/**: Stored model weights after training. 
- **App/**: Streamlit application scripts for the interactive user interface.


## Contact

For any queries, please raise an issue in the repository or contact at eklipinaa@gmail.com