# 🩺 Kidney Disease Detection from Medical Images

## 📌 Project Overview

This project focuses on automated kidney disease detection using medical imaging data (CT/MRI scans). A machine learning pipeline is designed to preprocess kidney images, extract meaningful features, and classify disease conditions using a trained model. The system is deployed as an interactive Streamlit web application for real-time inference.

## 🎯 Problem Statement

Manual analysis of medical images can be time-consuming and error-prone. This project aims to assist in the early identification of kidney-related conditions by leveraging machine learning techniques on medical image data.

## 🧠 Methodology

**Input Data:** Kidney CT/MRI images

**Preprocessing Steps:**

Conversion to grayscale

Image resizing to 64×64

Pixel normalization

Feature flattening

**Model Used:** Decision Tree Classifier

**Prediction Output:** Normal, Stone, Cyst, or Tumor

To prevent misuse, the application includes user warnings and confirmation prompts, ensuring that predictions are made only on kidney-related medical images.

## 🚀 Application Features

Upload and preview medical images

Real-time disease prediction

User confirmation for valid medical input

Clean and professional medical-themed interface

Streamlit-based deployment

## 🛠️ Technology Stack

Python

NumPy, OpenCV, Pillow

Scikit-learn

Streamlit

## ▶️ How to Run
pip install -r requirements.txt

streamlit run app.py

## ⚠️ Disclaimer

This application is developed strictly for educational and research purposes. It is not intended for clinical diagnosis or medical decision-making.

## ✅ Conclusion

This project demonstrates the practical application of machine learning in medical image analysis, covering the complete workflow from preprocessing to deployment. By integrating a classical ML model with a user-friendly interface, the system highlights how ML techniques can support healthcare-related problem statements while emphasizing responsible and informed usage.
