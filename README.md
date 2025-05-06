# 🌸 Flower Image Classifier - Train and Predict with Deep Learning

This project implements a deep learning image classifier to identify different species of flowers using a **pretrained VGG16 model**.  
It consists of two **command-line scripts**:

1. **`train.py`** - Trains a neural network on a dataset of flower images and saves the trained model.
2. **`predict.py`** - Loads a trained model and makes predictions on new images.

---

## 🚀 **Installation Instructions**
Before using this project, install the required dependencies:

```bash
pip install torch torchvision numpy matplotlib pillow argparse json5



### **📌 Summary of Added Commands**
| Action | Command |
|--------|---------|
| **Train on CPU** | `python train.py flowers` |
| **Train on GPU** | `python train.py flowers --gpu` |
| **Train with ResNet50** | `python train.py flowers --arch resnet50` |
| **Save model in specific directory** | `python train.py flowers --save_dir my_models` |
| **Predict an image** | `python predict.py flowers/test/19/image_06186.jpg saved_models/checkpoint_vgg16.pth` |
| **Predict top 3 classes** | `python predict.py flowers/test/19/image_06186.jpg saved_models/checkpoint_vgg16.pth --top_k 3` |
| **Predict on GPU** | `python predict.py flowers/test/19/image_06186.jpg saved_models/checkpoint_vgg16.pth --gpu` |
