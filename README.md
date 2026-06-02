# 🐾 FaunaVision ML: Deep Learning-Based Cat vs Dog Classifier

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.9+-blue.svg?style=for-the-badge&logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white" alt="PyTorch">
  <img src="https://img.shields.io/badge/Streamlit-%23FF4B4B.svg?style=for-the-badge&logo=Streamlit&logoColor=white" alt="Streamlit">
  <img src="https://img.shields.io/badge/Hugging%20Face-Spaces-%23FFD21E.svg?style=for-the-badge" alt="HuggingFace">
  <img src="https://img.shields.io/badge/License-GPLv3-green.svg?style=for-the-badge" alt="License">
</p>

---

### 🚀 Live Demonstration
🎯 **Experience the Web App Live:** [FaunaVision ML on Hugging Face Spaces](https://huggingface.co/spaces/bigganbiggan/FaunaVision-ML) 
### 📓 Training Environment
- **Kaggle Notebook Link:** https://www.kaggle.com/code/gmbiggan/deep-learning-classification-cat-dog


---

## 📌 Project Overview
**FaunaVision ML** is an end-to-end, production-ready computer vision application designed to classify images of cats and dogs with high precision. Powered by a custom-built **Convolutional Neural Network (CNN)** in **PyTorch**, the model is seamlessly integrated into an interactive web interface using **Streamlit** and hosted on **Hugging Face Spaces**.

Beyond basic classification, the system implements robust real-world optimization strategies, including an out-of-distribution (OOD) filter to safeguard against non-animal inputs.

---

## ✨ Key Features

* **🧠 Custom 4-Layer CNN Architecture:** Engineered from scratch with optimized hidden layers, utilizing **Batch Normalization**, **Max-Pooling**, and **Dropout** layers to rigorously combat overfitting and maintain generalize performance.
* **🎨 Premium Interactive UI:** A clean, fully customized Streamlit frontend enhanced with raw **HTML/CSS injection** supporting responsive dark/light aesthetic adaptions.
* **📸 Dual Media Input:** Supports local image file uploads (`.jpg`, `.jpeg`, `.png`) as well as real-time webcam/mobile camera captures directly through the browser.
* **🛡️ Safety Confidence Threshold:** Features an intelligent edge-case handler. If the model's prediction confidence drops below **80%**, it dynamically rejects the classification to filter out out-of-distribution or irrelevant images.

---

## 🛠️ Tech Stack & Architecture

### **Core Technologies**
* **Deep Learning Framework:** `PyTorch` (torch, torchvision)
* **Web Application Interface:** `Streamlit`
* **Data Manipulation & Logic:** `NumPy`, `PIL` (Pillow)
* **Model Tracking & Version Control:** `Git` / `GitHub CLI`

### **Model Specifications**
* **Dataset:** Kaggle Cats and Dogs Dataset (thousands of filtered biological images).
* **Input Pipeline:** Images are dynamically resized to $128 \times 128$ pixels, normalized, and converted to PyTorch tensors before entering the feedforward pipeline.
* **Loss Function & Optimizer:** Cross-Entropy Loss combined with the Adam optimizer for smooth gradient descent.

---

## 💻 Local Installation & Setup

If you want to run this project locally on your machine, follow these simple steps:

### **1. Clone the Repository**
```bash
git clone [https://github.com/Biggan2003/FaunaVision-ML-Deep-Learning-Based-Cat-vs-Dog-Classifier.git](https://github.com/Biggan2003/FaunaVision-ML-Deep-Learning-Based-Cat-vs-Dog-Classifier.git)
cd FaunaVision-ML-Deep-Learning-Based-Cat-vs-Dog-Classifier
```
---

## 🔑 Developer's Notebook ............
<details>
<summary><b>📌 Click here to view Project Credentials & Training Logs (Internal Use Only)</b></summary>

### 📓 Training Environment
- **Kaggle Notebook Link:** https://www.kaggle.com/code/gmbiggan/deep-learning-classification-cat-dog
- **Dataset Source:** https://www.kaggle.com/datasets/tongpython/cat-and-dog

### 🚀 Deployment Details (Hugging Face)
- **Registered Email:** gmb@gmail id theke register kora
- **Hugging Face Username:** gmb amr bektigoto id theke
- **Space Repository Link:** https://huggingface.co/spaces/bigganbiggan/FaunaVision-ML

</details>

