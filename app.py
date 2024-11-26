#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 25 22:48:52 2024

@author: laurajorgensen

cd /Users/laurajorgensen/Desktop/Desktop

streamlit run app.py

"""

import streamlit as st
import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import euclidean_distances
from PIL import Image, ExifTags

# Helper function to handle EXIF orientation
def correct_image_orientation(image):
    try:
        # Fetch EXIF data
        exif = image._getexif()
        if exif is not None:
            for tag, value in exif.items():
                if ExifTags.TAGS.get(tag) == 'Orientation':
                    if value == 3:  # Rotated 180 degrees
                        image = image.rotate(180, expand=True)
                    elif value == 6:  # Rotated 270 degrees
                        image = image.rotate(270, expand=True)
                    elif value == 8:  # Rotated 90 degrees
                        image = image.rotate(90, expand=True)
                    break
    except AttributeError:
        # If the image has no EXIF data, do nothing
        pass
    return image


# Helper function to preprocess images
def preprocess_image(image, size=(100, 100)):
    image = image.convert("L")  # Convert to grayscale
    image = image.resize(size)  # Resize to fixed size
    return np.array(image).flatten()

# Helper function to crop images to square
def crop_to_square(image):
    width, height = image.size
    if width == height:
        return image  # Already square
    else:
        new_size = min(width, height)
        left = (width - new_size) / 2
        top = (height - new_size) / 2
        right = (width + new_size) / 2
        bottom = (height + new_size) / 2
        return image.crop((left, top, right, bottom))

# Helper function to resize images for uniform display
def resize_image_for_display(image, new_size=(100, 100)):
    image = crop_to_square(image)
    return image.resize(new_size)

# Helper function to display images in a left-aligned grid layout
def display_images_in_grid(images, captions, max_per_row=5):
    rows = [images[i:i + max_per_row] for i in range(0, len(images), max_per_row)]
    captions_rows = [captions[i:i + max_per_row] for i in range(0, len(captions), max_per_row)]
    
    for row, captions_row in zip(rows, captions_rows):
        cols = st.columns(max_per_row)  # Always create a fixed number of columns per row
        for i, img in enumerate(row):
            with cols[i]:
                st.image(img, caption=captions_row[i], width=100)
        # Fill the remaining columns with empty space for alignment
        for i in range(len(row), max_per_row):
            cols[i].empty()
            
# App title and instructions
st.title("Ansigtsgenkendelse med Eigenface-metoden")
st.write("Upload mindst 3 billeder.")

# Information about HEIC files and conversion
st.markdown(
    """
    **Bemærk:** HEIC-filer kan ikke uploades. Enten konverter dem lokalt på din computere ellers ændre formateringsindstillingen i din mobil inden billederne bliver taget. Alternativt brug en online-konverter som [heictojpg.com](https://heictojpg.com).
    """
)

# Step 1: Upload training images
st.header("1. Upload træningsbilleder")
uploaded_files = st.file_uploader(
    "Upload mindst 3 billeder (støttede formater: .png, .jpg, .jpeg)",
    accept_multiple_files=True,
    type=["png", "jpg", "jpeg"]
)

if uploaded_files:
    if len(uploaded_files) < 3:
        st.warning("Du skal uploade mindst 3 billeder for at fortsætte.")
    else:
        #st.write("Billederne er uploadet. Vi forbereder dem til analyse...")
        training_images = []
        labels = []
        resized_images = []  # For storing resized images

        # Update your image processing
        for i, file in enumerate(uploaded_files):
            img = Image.open(file)
            img = correct_image_orientation(img)  # Correct EXIF orientation
            resized_img = resize_image_for_display(img)  # Crop and resize for uniform display
            resized_images.append(resized_img)
            training_images.append(preprocess_image(img))  # Preprocess for PCA
            labels.append(f"Billede {i+1}")

        training_images = np.array(training_images)
        st.write(f"{len(training_images)} billeder er oploadet.")

        # Display resized versions of the uploaded images in a grid
        st.subheader("Uploadede billeder:")
        display_images_in_grid(resized_images, captions=labels)

        # Step 2: Calculate eigenfaces
        st.header("2. Generer eigenfaces")
        max_components = min(len(training_images), training_images.shape[1])  # Min of samples and features
        n_components = st.slider("Vælg antal eigenfaces:", 1, max_components, min(3, max_components))
        pca = PCA(n_components=n_components)
        pca.fit(training_images)
        eigenfaces = pca.components_

        # Visualize eigenfaces in a grid
        st.subheader("Eigenfaces:")
        eigenface_images = []
        eigenface_captions = []
        for i, eigenface in enumerate(eigenfaces):
            normalized_eigenface = np.interp(eigenface, (eigenface.min(), eigenface.max()), (0, 255))
            resized_eigenface = normalized_eigenface.reshape(100, 100)
            eigenface_images.append(resized_eigenface.astype(np.uint8))
            eigenface_captions.append(f"Eigenface {i+1}")

        display_images_in_grid(eigenface_images, captions=eigenface_captions)

        # Step 3: Test recognition
        st.header("3. Ansigtsgenkendelses test")
        test_file = st.file_uploader(
            "Upload et billede for at teste genkendelse (støttede formater: .png, .jpg, .jpeg)",
            type=["png", "jpg", "jpeg"]
        )
        
        if test_file:
            # Åbn testbillede uden rotation for beregninger
            test_img = Image.open(test_file)
            test_img_preprocessed = preprocess_image(test_img)  # Brug originalt billede til beregning
        
            # Rotér testbilledet KUN til visning
            test_img_corrected = correct_image_orientation(test_img)
            test_img_resized_for_display = resize_image_for_display(test_img_corrected)
        
            # Beregn gennemsnit og centrer træningsbilleder
            mean_face = np.mean(training_images, axis=0)
            centered_training = training_images - mean_face
        
            # Brug kun de valgte eigenfaces til beregninger
            selected_eigenfaces = eigenfaces[:n_components]  # Begræns til det valgte antal eigenfaces
        
            # Projekter træningsbilleder på eigenfaces
            training_projections = np.dot(centered_training, selected_eigenfaces.T)
        
            # Projekter testbillede på eigenfaces
            centered_test_img = test_img_preprocessed - mean_face
            test_projection = np.dot(selected_eigenfaces, centered_test_img)
        
            # Sammenlign afstande mellem testbillede og træningsbilleder
            distances = euclidean_distances([test_projection], training_projections)
            closest_idx = np.argmin(distances)
        
            # Find matchet billede uden rotation for beregning
            matched_img = Image.open(uploaded_files[closest_idx])
            matched_img_corrected = correct_image_orientation(matched_img)  # Rotér kun til visning
        
            # Visning af billeder (roterede)
            result_images = [test_img_resized_for_display, resize_image_for_display(matched_img_corrected)]
            result_captions = ["Testbillede", f"Matchet billede: {labels[closest_idx]}"]
        
            # Vis testbillede og matchet billede i grid
            st.subheader("Resultater:")
            cols = st.columns(5)
            for i, (img, caption) in enumerate(zip(result_images, result_captions)):
                with cols[i]:
                    st.image(img, caption=caption, width=100)