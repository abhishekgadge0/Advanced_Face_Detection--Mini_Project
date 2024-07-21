# Advanced Face Detection System

## Introduction
This mini project, developed during my sixth semester in college, is an advanced face detection system that not only identifies individuals but also predicts their age and gender. The project leverages computer vision and deep learning techniques to achieve accurate predictions.

## Project Goals
- Capture and store images of faces.
- Train a face recognizer model.
- Recognize faces in real-time.
- Predict the age and gender of identified individuals.
- Provide functionality to delete captured data when needed.

## Methodology
- **Face Detection and Image Capture:** Capture images of faces using a webcam, detect faces in the images, and store the images for training.
- **Model Training:** Train a Local Binary Patterns Histograms (LBPH) face recognizer model using the captured images.
- **Age and Gender Prediction:** Use pre-trained deep learning models to predict the age and gender of identified individuals.
- **Data Management:** Provide options to delete all captured data or data for a specific individual.

## Dependencies
- Python
- OpenCV
- NumPy

## Dataset
The project utilizes the following pre-trained models for age and gender prediction:
- `gender_net.caffemodel`
- `gender_deploy.prototxt`
- `age_net.caffemodel`
- `age_deploy.prototxt`

Ensure these model files are placed in the correct directories as specified in the code.

## Contribution
Contributions are welcome! Feel free to create issues, suggest improvements, or submit pull requests.

Let's work together to enhance face detection systems with advanced age and gender prediction capabilities!
