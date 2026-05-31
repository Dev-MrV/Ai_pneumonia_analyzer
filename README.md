# 🩺 AI-Based Pneumonia Detection Using Chest X-Ray Images

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)
![FastAI](https://img.shields.io/badge/FastAI-Deep%20Learning-red)
![PyTorch](https://img.shields.io/badge/PyTorch-Framework-orange?logo=pytorch)
![ResNet50](https://img.shields.io/badge/Model-ResNet50-success)
![Streamlit](https://img.shields.io/badge/Streamlit-Web%20App-FF4B4B?logo=streamlit)
![Computer Vision](https://img.shields.io/badge/Computer%20Vision-Medical%20Imaging-green)
![Healthcare AI](https://img.shields.io/badge/Healthcare-AI-blueviolet)
![License](https://img.shields.io/badge/License-MIT-yellow)
![Status](https://img.shields.io/badge/Project-Completed-brightgreen)
![Dataset](https://img.shields.io/badge/Dataset-Kaggle-blue)
![Accuracy](https://img.shields.io/badge/Validation%20Accuracy-75%25-success)
![Made With](https://img.shields.io/badge/Made%20with-❤️-red)

## 📌 Overview

Pneumonia is a serious lung infection that can be life-threatening if not diagnosed and treated early. This project leverages Deep Learning and Computer Vision techniques to automatically detect pneumonia from chest X-ray images.


The system uses a pre-trained **ResNet50** Convolutional Neural Network (CNN) with **Transfer Learning** through the **FastAI** framework to classify chest X-ray images into:

* ✅ Normal
* ⚠️ Pneumonia

A user-friendly **Streamlit web application** was developed to allow users to upload chest X-ray images and receive instant AI-powered predictions with confidence scores.

---

## 🚀 Features

* Chest X-ray image classification
* Transfer Learning using ResNet50
* FastAI-based model training
* Interactive Streamlit web application
* Image upload and preview
* Real-time prediction results
* Confidence score visualization
* Medical AI project for healthcare applications

---

## 🛠️ Technologies Used

### Programming Language

* Python

### Deep Learning Frameworks

* FastAI
* PyTorch

### Web Framework

* Streamlit

### Libraries

* NumPy
* Matplotlib
* Pillow

### Development Environment

* Google Colab
* Jupyter Notebook

### Version Control

* Git
* GitHub

---

## 📂 Dataset

### Dataset Name

Chest X-Ray Images (Pneumonia)

### Source

Kaggle

### Dataset Link

https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia

### Classes

* Normal
* Pneumonia

### Dataset Structure

```text
chest_xray/
│
├── train/
│   ├── NORMAL/
│   └── PNEUMONIA/
│
├── val/
│   ├── NORMAL/
│   └── PNEUMONIA/
│
└── test/
    ├── NORMAL/
    └── PNEUMONIA/
```

---

## 🔄 Project Workflow

```text
Dataset Collection
        │
        ▼
Data Preprocessing
        │
        ▼
Data Augmentation
        │
        ▼
ResNet50 Model Loading
        │
        ▼
Transfer Learning
        │
        ▼
Model Training
        │
        ▼
Model Evaluation
        │
        ▼
Model Export
        │
        ▼
Streamlit Deployment
        │
        ▼
Disease Prediction
```

---

## 🧠 Model Architecture

### ResNet50

ResNet50 is a deep Convolutional Neural Network consisting of 50 layers and residual connections that help overcome the vanishing gradient problem.

### Why ResNet50?

* High image classification accuracy
* Efficient transfer learning
* Reduced training time
* Excellent feature extraction capabilities
* Widely used in medical imaging applications

---

## ⚙️ Installation

### Clone Repository

```bash
git clone https://github.com/your-username/pneumonia-detection.git
cd pneumonia-detection
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 📦 Requirements

```txt
streamlit
fastai
torch
torchvision
numpy
matplotlib
pillow
```

---

## 🏋️ Training the Model

### Load Dataset

```python
from fastai.vision.all import *

path = Path('/content/chest_xray')

dls = ImageDataLoaders.from_folder(
    path,
    train='train',
    valid='val',
    item_tfms=Resize(224),
    batch_tfms=aug_transforms(),
    bs=32
)
```

### Create Learner

```python
learn = cnn_learner(
    dls,
    resnet50,
    metrics=accuracy
)
```

### Train Model

```python
learn.fine_tune(5)
```

### Export Model

```python
learn.export('pneumonia_model.pkl')
```

---

## 🖥️ Running the Streamlit App

```bash
streamlit run app.py
```

The application will launch locally at:

```text
http://localhost:8501
```

---

## 📊 Results

| Metric              | Value  |
| ------------------- | ------ |
| Validation Accuracy | 75.00% |
| Training Loss       | 0.0784 |
| Validation Loss     | 0.4195 |

### Key Outcomes

* Successful pneumonia detection from chest X-ray images.
* Transfer learning reduced training time.
* Developed an interactive AI-powered healthcare application.
* Demonstrated practical use of computer vision in medical diagnosis.

---

## 📸 Project Screenshots
![](https://github.com/Dev-MrV/Ai_pneumonia_analyzer/blob/main/Screenshots/image3.png)
![](https://github.com/Dev-MrV/Ai_pneumonia_analyzer/blob/main/Screenshots/image4.png)
![](https://github.com/Dev-MrV/Ai_pneumonia_analyzer/blob/main/Screenshots/image4.png)
![](https://github.com/Dev-MrV/Ai_pneumonia_analyzer/blob/main/Screenshots/image5.png)
![](https://github.com/Dev-MrV/Ai_pneumonia_analyzer/blob/main/Screenshots/image6.png)
![](https://github.com/Dev-MrV/Ai_pneumonia_analyzer/blob/main/Screenshots/image7.png)
---

## ⚠️ Challenges Faced

* Managing large image datasets in Google Colab.
* Limited validation dataset size.
* Model compatibility issues between Colab and local environments.
* Debugging Streamlit deployment errors.
* Optimizing model performance and prediction accuracy.

---

## 🔮 Future Enhancements

* Improve model accuracy using larger datasets.
* Detect additional lung diseases.
* Integrate Grad-CAM visualization.
* Cloud deployment for public access.
* PDF report generation.
* Mobile application development.
* Patient management dashboard.

---

## 📚 References

### Dataset

* https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia

### Model Used

* https://huggingface.co/devmrv/ai_pneumonia_analyzer

### Documentation

* https://docs.fast.ai
* https://pytorch.org/docs
* https://docs.streamlit.io

### Research Paper

* Deep Residual Learning for Image Recognition (ResNet50)

---

## 👨‍💻 Author

**DevMrV**

AI/ML Mini Project

Computer Science and Engineering

---

## ⭐ Acknowledgement

Special thanks to the FastAI, PyTorch, Streamlit, and Kaggle communities for providing open-source tools, datasets, and learning resources that made this project possible.
