# 🧠 Comprehensive CNN Models for Medical Image Classification

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)
![Keras](https://img.shields.io/badge/Keras-Deep%20Learning-red)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/satyajeetparmar0/image-classification-cnn/blob/main/simple_cnn.ipynb)


This project implements **7 state-of-the-art CNN architectures** for binary medical image classification, specifically designed to classify cancer images into **Benign** and **Malignant** categories.

---

## 🎯 **Implemented Models**

| Model | Architecture | Status | Key Features |
|-------|-------------|--------|--------------|
| ✅ **Simple CNN** | Sequential CNN | Complete | Basic 2-layer architecture |
| ✅ **VGG16** | VGG16-like | Complete | Deep sequential blocks |
| ✅ **ResNet** | Residual Network | Complete | Skip connections, residual blocks |
| ✅ **DenseNet** | Dense Convolutional | Complete | Dense connections, feature reuse |
| ✅ **EfficientNet** | Efficient CNN | Complete | MBConv blocks, compound scaling |
| ✅ **GoogLeNet** | Inception Network | Complete | Inception blocks, parallel paths |
| ✅ **ConvNeXt** | Modern CNN | Complete | Layer normalization, modern design |

---

## 📁 **Project Structure**

```
CNN/
├── simple_cnn.ipynb
├── vgg16_model.ipynb
├── resnet.ipynb
├── densenet.ipynb
├── efficientnet.ipynb
├── googlenet.ipynb
├── convnext.ipynb
├── README.md
├── requirements.txt
└── dataset_info.txt

```

---

## ⚡ **Key Features**

### 🧠 **Model Diversity**
- **7 different CNN architectures** from simple to complex
- **State-of-the-art implementations** following research papers
- **Consistent code structure** across all models
- **Binary classification** optimized for medical images

### 🔧 **Technical Excellence**
- **Google Colab compatible** notebooks
- **Reproducible results** with fixed random seeds
- **Error handling** and robust training
- **Modular design** for easy customization

---

## 🛠 **Tech Stack**

- **Deep Learning**: TensorFlow/Keras
- **Data Processing**: NumPy, scikit-learn
- **Visualization**: Matplotlib (optional)
- **Environment**: Google Colab / Jupyter Notebook
- **Language**: Python 3.8+

---

## 🚀 **Quick Start**
```bash
# Open any model notebook in Google Colab
# Example: simple_cnn.ipynb, vgg16_model.ipynb, etc.
```

---

## 📂 **Dataset Requirements**

### **Structure**
```
dataset_cancer_v1/
└── classificacao_binaria/
    └── 100X/
        ├── benign/
        │   ├── img001.png
        │   ├── img002.png
        │   └── ...
        └── malignant/
            ├── img001.png
            ├── img002.png
            └── ...
```

### **Specifications**
- **Format**: PNG, JPG, JPEG
- **Size**: Automatically resized to 224x224
- **Classes**: Binary (Benign vs Malignant)
- **Path**: `/content/drive/MyDrive/dataset_cancer_v1/classificacao_binaria/100X`

---

## 📊 **Performance Metrics**

Each model is evaluated on:
- **Test Accuracy** - Final classification accuracy
- **Precision** - True positive rate
- **Recall** - Sensitivity
- **F1-Score** - Harmonic mean of precision and recall
- **Training Time** - Computational efficiency
- **Model Parameters** - Complexity comparison

---

## 🎯 **Model Architectures Overview**

### **Simple CNN**
- 2 convolutional layers with max pooling
- Dense layers with dropout
- Lightweight and fast training

### **VGG16**
- 5 convolutional blocks
- Deep sequential architecture
- Proven performance on medical images

### **ResNet**
- Residual connections
- Batch normalization
- Addresses vanishing gradient problem

### **DenseNet**
- Dense connections between layers
- Feature reuse and gradient flow
- Efficient parameter usage

### **EfficientNet**
- MBConv blocks with squeeze-and-excitation
- Compound scaling method
- Optimal accuracy-efficiency trade-off

### **GoogLeNet**
- Inception blocks with parallel paths
- Multiple filter sizes (1x1, 3x3, 5x5)
- Auxiliary classifiers

### **ConvNeXt**
- Modern design inspired by Vision Transformers
- Layer normalization and GELU activation
- State-of-the-art performance

---

## 🔧 **Configuration**

All models use consistent hyperparameters:
```python
IMG_SIZE = (224, 224)      # Image dimensions
EPOCHS = 10               # Training epochs
BATCH_SIZE = 32           # Batch size
LEARNING_RATE = 0.0001    # Learning rate
VALIDATION_SPLIT = 0.2    # Validation ratio
```

---

## 📈 **Expected Results**

Based on typical performance:
- **Simple CNN**: ~80-85% accuracy
- **VGG16**: ~85-90% accuracy
- **ResNet**: ~88-92% accuracy
- **DenseNet**: ~90-93% accuracy
- **EfficientNet**: ~91-94% accuracy
- **GoogLeNet**: ~89-92% accuracy
- **ConvNeXt**: ~92-95% accuracy

*Results may vary based on dataset quality and training conditions*

---

## 🐛 **Troubleshooting**

### **Common Issues**
1. **Memory Errors**: Reduce batch size or image size
2. **Long Training**: Reduce epochs or use simpler models
3. **Low Accuracy**: Check dataset quality and preprocessing
4. **File Not Found**: Verify dataset path structure

---

## 🤝 **Contributing**

1. Fork the repository
2. Create a feature branch
3. Implement your changes
4. Test thoroughly
5. Submit a pull request

---

## 👨‍💻 **Author**

**Satyajeet Parmar**  
🔗 [LinkedIn](https://linkedin.com/in/satyajeet-parmar/) | 🐙 [GitHub](https://github.com/satyajeetparmar0)

---

## 🙏 **Acknowledgments**

- Medical image dataset providers
- TensorFlow/Keras community
- Research papers and implementations referenced
- Open source contributors

---

## 📞 **Support**

For questions, issues, or contributions:
- 📧 Create an issue on GitHub
- 💬 Contact via LinkedIn
- 📖 Check the documentation files

---

**⭐ Star this repository if you find it helpful!**



