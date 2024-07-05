from Datasets.utils import extract_and_save_features



image_folder = 'sketches'  # Folder containing your images
output_folder = 'features_sketches'  # Folder to save the extracted features
extract_and_save_features(image_folder, output_folder)

