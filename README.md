# Bladder Cancer Detection Project

This project is a collection of Python scripts for bladder cancer detection and classification in medical images and videos. It utilizes deep learning models built with TensorFlow, Keras, and TensorFlow Lite, along with image processing techniques using OpenCV.

## Files

### `bladder_type.py`
This script is likely used for classifying bladder types. (More details would be needed from the code to be specific).

### `contour.py`
This script probably contains functions for finding and drawing contours of objects in images, a common task in image analysis.

### `extract_frame.py`
This script is used to extract individual frames from a video file. This is a common preprocessing step for video analysis.

### `from_json_to_xml.py`
This script converts annotations from JSON format to XML format. This is often necessary to make datasets compatible with different annotation tools or deep learning frameworks.

### `object_detection_tensorflowlite.py`
This is one of the main scripts of the project. It performs object detection on a video using a pre-trained TensorFlow Lite model (`detect.tflite`). It loads a video, processes each frame, runs inference to detect objects, and then draws bounding boxes around the detected objects in the video stream.

**Note:** The path to the video file is hardcoded in this script. You will need to change the `VIDEO_NAME` variable to the path of your video file.

### `object_detection_video.py`
This script performs video classification using a Keras model (`bladder_types.h5`). It classifies each frame of a video into one of the following categories: 'NMIBC', 'Normal', or 'Ta'.

**Note:** The path to the video file is hardcoded in this script. You will need to change the path in the `cv2.VideoCapture()` function call.

### `panorama.py`
This script is likely used to stitch multiple images together to create a panoramic image.

### `randomly_distribute_data.py`
This script is a utility for randomly distributing data, likely for creating training and testing sets.

### `save_images_and_annotations_in_desired_structure.py`
This script helps in organizing image and annotation files into a specific directory structure, which is often required for training deep learning models.

### `split_dataset_into_training_validation_testing.py`
This script splits a dataset into training, validation, and testing sets, which is a crucial step in any machine learning project.

### `thresholding.py`
This script is used for applying image thresholding, a technique to segment an image into a binary image.

### `thresholding_with_model.py`
This script likely combines image thresholding with the output of a deep learning model.

## `Sample_TFLite_model` Directory

This directory contains the TensorFlow Lite model and associated files.

*   `detect.tflite`: This is the TensorFlow Lite model file used for object detection.
*   `labelmap.txt`: This file contains the labels for the objects that the model can detect. In this case, it contains "Suspected_Regain".

## Setup and Usage

1.  **Dependencies:**
    While a `requirements.txt` file is not provided, you will need to install the following Python libraries:
    *   OpenCV: `pip install opencv-python`
    *   NumPy: `pip install numpy`
    *   TensorFlow: `pip install tensorflow`
    *   Keras: `pip install keras` (usually comes with TensorFlow)

2.  **Running the scripts:**
    *   To run the object detection script, navigate to the project directory and run:
        ```bash
        python object_detection_tensorflowlite.py
        ```
        Remember to change the hardcoded video path inside the script.
    *   To run the video classification script, run:
        ```bash
        python object_detection_video.py
        ```
        Remember to change the hardcoded video path inside the script.

## Project Overview

The main goal of this project is to provide tools for analyzing medical imaging data related to bladder cancer. It includes scripts for both object detection (to locate specific regions of interest, like "Suspected_Regain") and classification (to categorize bladder conditions like 'NMIBC', 'Normal', 'Ta'). The various utility scripts help in preparing and managing the data for these tasks.
