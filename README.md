# Face Emotion Detection

This project focuses on detecting human emotions from facial images using **Convolutional Neural Networks (CNNs)**. It explores how deep learning can automatically extract features from faces to classify emotions accurately.

## Overview

Face Emotion Detection is a vital task in computer vision and artificial intelligence. It involves recognizing emotional states such as **happy, sad, angry, surprised, neutral**, and more from facial expressions. Applications include:

* Human-computer interaction
* Sentiment analysis
* Healthcare and therapy
* Security and surveillance

This project implements a **CNN-based approach**, which excels at learning hierarchical features from images.

## Problem Statement

Traditional machine learning approaches require manual feature extraction (like edge detection, HOG, or PCA). CNNs, on the other hand, automatically learn relevant features from raw images, making them more efficient and accurate for facial emotion recognition.

## Methodology

1. **Data Preprocessing**

   * Images are converted to grayscale to reduce complexity.
   * Normalization scales pixel values between 0 and 1.
   * Images are resized to a fixed dimension to maintain consistency.

2. **CNN Architecture**

   * **Convolutional Layers**: Extract spatial features using filters.
   * **Activation Functions (ReLU)**: Introduce non-linearity.
   * **Pooling Layers (Max Pooling)**: Reduce dimensionality while retaining key features.
   * **Fully Connected Layers**: Map learned features to emotion classes.
   * **Softmax Layer**: Produces probability distribution over emotion classes.

3. **Training**

   * Loss Function: Categorical Crossentropy
   * Optimizer: Adam or SGD
   * Metrics: Accuracy and loss monitoring

4. **Evaluation**

   * The model is evaluated on unseen test data to measure its accuracy in predicting emotions.
   * Confusion matrices and classification reports are used for detailed performance analysis.

## Key Advantages

* **Automatic Feature Extraction**: No need for manual feature engineering.
* **High Accuracy**: CNNs can capture complex facial patterns effectively.
* **Real-Time Prediction**: Can be extended to live webcam feeds for instant emotion detection.

## Conclusion

This CNN-based Face Emotion Detection system demonstrates how deep learning models can be applied to human emotion recognition. It forms a foundation for further research in affective computing and human-centric AI applications.


# Emotion_Detection_CNN

Data Set Link - https://www.kaggle.com/jonathanoheix/face-expression-recognition-dataset
