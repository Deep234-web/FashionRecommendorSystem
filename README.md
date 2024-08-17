# Fashion Recommender System

## Overview

This project is an image-based fashion recommender system that uses deep learning to extract features from images and recommend similar items based on visual similarity. The core idea is to allow users to upload an image, extract its features using a pre-trained ResNet50 model, and then find and display similar images from a precomputed dataset.

### Features
- Upload an image to get recommendations for similar images.
- Uses ResNet50, a popular deep learning model, for feature extraction.
- Finds similar images using the Nearest Neighbors algorithm with Euclidean distance.

### Technologies Used
- **TensorFlow/Keras**: For loading the pre-trained ResNet50 model and extracting features.
- **scikit-learn**: For the Nearest Neighbors algorithm to find similar images.
- **Streamlit**: For creating the web interface to upload images and display recommendations.
- **OpenCV**: For handling and displaying images during testing (optional).

## Project Structure

- `app.py`: This script loads the images from a directory, extracts their features using ResNet50, and stores the features and filenames as pickle files for later use.
- `main.py`: The main script that runs the Streamlit app, allowing users to upload images and get recommendations for similar images.
- `test.py`: A script for testing the recommendation system outside of the Streamlit interface.
- `uploads/`: Directory where uploaded images are stored temporarily during processing.
- `images/`: Directory where your image dataset is stored (not included in the repository due to size constraints).
- `embeddings.pkl`: Precomputed feature vectors of the images (not included in the repository due to size constraints).
- `filenames.pkl`: List of filenames corresponding to the images in your dataset (not included in the repository due to size constraints).

## Important Note

### Dataset and Pickle Files
Due to storage constraints, the image dataset and the pickle files (`embeddings.pkl` and `filenames.pkl`) are **not included** in this repository. These files total around **25GB**, which makes them too large to upload to a typical version control platform like GitHub.
