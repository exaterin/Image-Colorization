import cv2
import numpy as np

# Read the image
image = cv2.imread('Dataset/Images/Q1389_wd0.jpg')

# Convert the image to Lab colorspace
lab_image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)

ab_channels = lab_image[:, :, 1:]

# Quantize ab channels with the step 10
ab_channels = (ab_channels // 10) * 10

# Show the quantized image
quantized_image = np.zeros_like(lab_image)
quantized_image[:, :, 0] = lab_image[:, :, 0]
quantized_image[:, :, 1:] = ab_channels
quantized_image = cv2.cvtColor(quantized_image, cv2.COLOR_LAB2BGR)
cv2.imshow('Quantized Image', quantized_image)

# print the quantized ab channels
print(ab_channels)

# Print all unique tuples (a, b) in the quantized ab channels
print(len(np.unique(ab_channels.reshape(-1, 2), axis=0)))