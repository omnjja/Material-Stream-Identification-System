# MSI-System

**Project:** Real-Time Material Classification System using Machine Learning  
**Course:** CS462-Machine Learning

**Date:** 18, December, 2025

---
## Project Overview

This project implements a **real-time waste classification system** using machine learning.  
The system classifies waste into the following categories:

- Glass, Paper, Cardboard, Plastic, Metal, Trash, Unknown

The pipeline includes:

1. Data preparation and augmentation  
2. Feature extraction (HOG, LBP, Color & HSV histograms)  
3. Training SVM and K-NN classifiers  
4. Real-time classification using a live camera feed  

---

## Data Preparation

- Original images are stored in `dataset/`  
- Augmented images are stored in `augmented_dataset/`  
- Use `augment.py` and `feature_extraction.py` to prepare datasets for training  

---

## Model Training

- `train.py` trains SVM and K-NN classifiers  
- Models are saved in `models/` (`svm_model.pkl`, `knn_model.pkl`, `scaler.pkl`, `pca.pkl`)  

---

## Real-Time Classification

- Run `realtime_classification.py` to classify waste using a live camera feed  

---

## Testing & Accuracy

- `test.py` evaluates the model accuracy:  

```python
from test import predict, print_accuracy

preds = predict("test_data", "models/svm_model.pkl")
print_accuracy(y_true, preds)

