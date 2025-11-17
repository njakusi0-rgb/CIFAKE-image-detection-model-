# CIFAKE Image Classification: Real vs AI-Generated Images


A deep learning project that distinguishes between real photographs and AI-generated synthetic images using a Convolutional Neural Network (CNN) trained on the CIFAKE dataset.

Project Overview
This project implements a binary classification system that can differentiate between:

REAL images from the CIFAR-10 dataset

FAKE AI-generated synthetic images created by Stable Diffusion

The model achieves high accuracy in detecting AI-generated content, which has important applications in digital forensics, content authentication, and misinformation detection

Features
Deep Learning Model: CNN architecture with convolutional and pooling layers for feature extraction

Data Preprocessing: Automated image loading and normalization

Visualization Tools: Sample image display and training progress plots

Prediction Interface: Easy-to-use function for classifying new images

High Performance: Optimized for 32x32 RGB images with efficient batch processing

Installation & Setup

Prerequisites
pip install tensorflow matplotlib pillow kagglehub numpy

Dataset
The project uses the CIFAKE dataset from Kaggle, which is automatically downloaded:

import kagglehub
path = kagglehub.dataset_download("birdy654/cifake-real-and-ai-generated-synthetic-images")

Model Architecture
model = models.Sequential([
    # Feature Extraction Layers
    layers.Conv2D(32, (3,3), activation='relu', input_shape=(32, 32, 3)),
    layers.MaxPooling2D((2,2)),
    
    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D((2,2)),
    
    layers.Conv2D(64, (3,3), activation='relu'),
    
    # Classification Layers
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(1, activation='sigmoid')  # Binary output
])

Training
The model is trained with:

Optimizer: Adam

Loss Function: Binary Crossentropy

Metrics: Accuracy

Epochs: 10

Batch Size: 32

history = model.fit(
    train_generator,
    epochs=10,
    validation_data=test_generator
)

Usage
Classify New Images

# Predict on a single image
predict_new_image(model, "path_to_your_image.jpg")

# Output example:
# Image Path: path_to_your_image.jpg
# Predicted Class: REAL
# Confidence: 92.45%

Evaluate Model Performance
test_loss, test_acc = model.evaluate(test_generator)
print(f'Test accuracy: {test_acc:.2f}')

Results
The model provides:

High classification accuracy on test data

Confidence scores for predictions

Training/validation accuracy visualization

Real-time image classification capability

Applications
Content Moderation: Detect AI-generated images on social platforms

Digital Forensics: Authenticate image sources

Academic Research: Study AI image generation capabilities

Security: Identify synthetic media in critical contexts

Project Structure
├── cifake_classification.py    # Main implementation
├── requirements.txt           # Dependencies
├── models/                    # Saved model files
├── samples/                   # Test images
└── README.md                 # This file


Contributing
Contributions are welcome! Please feel free to submit pull requests, report bugs, or suggest new features.

License
This project is open source and available under the MIT License.

Acknowledgments
CIFAKE dataset creators

TensorFlow and Keras teams

Kaggle for hosting the dataset

Note: This project is for educational and research purposes. The effectiveness of AI-generated image detection may vary with different generative models and image qualities.

