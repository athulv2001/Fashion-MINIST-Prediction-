# 👕 Fashion MNIST Classification with CNN and Dropout

A deep learning project using Convolutional Neural Networks (CNNs) to classify images from the Fashion MNIST dataset. Achieved **93% test accuracy** after implementing Dropout layers to reduce overfitting.

## 🧠 Project Overview

Fashion MNIST is a dataset of Zalando's article images — a benchmark for machine learning algorithms, intended as a more realistic alternative to MNIST digits. Each image is a 28x28 grayscale image labeled with one of 10 clothing classes.

This project covers:
- Data exploration and visualization
- CNN model creation using Keras
- Training with and without Dropout
- Evaluation using accuracy, loss, and classification report
- Visualization of classified and misclassified samples

## 📊 Dataset

- 📁 Train: 60,000 images
- 📁 Test: 10,000 images
- 🔟 Classes: T-shirt/top, Trouser, Pullover, Dress, Coat, Sandal, Shirt, Sneaker, Bag, Ankle boot
- 🔲 Input shape: (28, 28, 1)

## 🧱 Model Architecture

1. Conv2D → ReLU → MaxPooling2D  
2. Conv2D → ReLU → MaxPooling2D  
3. Conv2D → ReLU  
4. Flatten  
5. Dense(128) → ReLU  
6. Dense(10) → Softmax

### 🛡️ Final Model (with Dropout)

Dropout layers added after pooling and dense layers:
- Dropout(0.25), Dropout(0.4), Dropout(0.3)

## 🚀 Results

| Metric           | Value     |
|------------------|-----------|
| ✅ Test Accuracy | ~93%      |
| ❌ Test Loss     | ~0.23     |

- Best classified classes: **Sneakers, Sandals, Ankle Boots**
- Most confused class: **Shirt (Class 6)**

## 📈 Visualization

- Class distribution plots
- Sample input images (train/test)
- Accuracy/Loss over epochs
- Correct vs Incorrect predictions with visualizations

## 📚 References

- [Fashion MNIST Dataset](https://www.kaggle.com/zalando-research/fashionmnist)
- [Keras Documentation](https://keras.io/)
- Dropout Theory: [Medium Article](https://medium.com/@vivek.yadav/why-dropouts-prevent-overfitting-in-deep-neural-networks-937e2543a701)

## 🛠 Tech Stack

- Python 🐍
- TensorFlow / Keras 📦
- Matplotlib / Seaborn / Plotly 📊
- Pandas / NumPy

---

## 📂 How to Run

```bash
git clone https://github.com/yourusername/fashion-mnist-cnn.git
cd fashion-mnist-cnn
pip install -r requirements.txt  # (Include requirements.txt if needed)
