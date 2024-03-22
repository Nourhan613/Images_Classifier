# Product Image Classifier

This repository contains code for training a deep-learning model to classify product images into four categories: accessories, fashion, games, and home.

## Dataset

The dataset used for training the model is a screenshot of the product images through this application: slash-eg.com.

## Model Architecture

The model architecture is based on a pre-trained VGG16 convolutional neural network, with additional layers added on top for fine-tuning.
The model is trained using TensorFlow and Keras.

## Data Preprocessing

The images are preprocessed by normalizing pixel values and resizing them to a common size (256x256 pixels).

## Training

The model is trained using a subset of the dataset, with a split of 60% for training, 20% for validation, and 20% for testing. Early stopping is employed to prevent overfitting.

## Evaluation

The model is evaluated on the training and validation sets using accuracy as the primary metric.

The evaluation results are as follows:
- Training Accuracy: [1.0]
- Validation Accuracy: [1.0]
- Testing Accuracy: [0.6666666865348816]

## Prediction

To make predictions on new images, the trained model can be used. 
Simply provide the path to the image file, and the model will predict the corresponding class.
