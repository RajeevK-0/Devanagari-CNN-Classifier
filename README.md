# Devanagari Script Character Recognition 🕉️

A Deep Learning project comparing the performance of Multi-Layer Perceptrons (MLP) and Convolutional Neural Networks (CNN) in classifying Devanagari handwritten characters.

## 📌 Project Overview
This project focuses on the classification of the Devanagari script (used for Hindi, Sanskrit, Marathi, etc.). The dataset consists of 32x32 pixel grayscale images across 46 distinct classes (36 consonants and 10 numerals). The goal is to build a robust pipeline that processes raw image data and accurately predicts the character class.

This repository demonstrates:
- **Data Engineering:** Custom pipeline to load, label, and preprocess image data from local directories.
- **Model Comparison:** Implementation of both a Standard ANN (Dense) and a custom CNN architecture.
- **Performance:** Achieving high accuracy using Convolutional layers to capture spatial hierarchies in character strokes.

## 🛠️ Tech Stack
* **Language:** Python 3.x
* **Deep Learning:** TensorFlow, Keras
* **Computer Vision:** OpenCV (`cv2`)
* **Data Manipulation:** NumPy
* **Visualization:** Matplotlib

## 📂 Dataset Structure
The model expects the data to be organized in the following directory structure:
```text
/dataset
    /train
        /character_1
        /character_2
        ...
    /test
        /character_1
        /character_2
        ...
