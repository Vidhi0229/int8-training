# MNIST CNN Model: Training, Quantization, and Evaluation

This project contains two Python scripts for training a Convolutional Neural Network (CNN) on the MNIST dataset, quantizing the model, and evaluating its performance.

## Files

1. `model.py`: Trains a CNN on MNIST data and quantizes the model to INT8.
2. `evaluate.py`: Evaluates the performance of the quantized INT8 model.

## Functionality

### model.py

This script:
- Defines a CNN architecture for MNIST classification
- Loads and preprocesses the MNIST dataset
- Trains the model on the training data
- Evaluates the model on the test data
- Saves the trained model in full precision (FP32) format
- Converts the model to TensorFlow Lite INT8 format
- Saves the quantized model as 'cnn_int8_model.tflite'

### evaluate.py

This script:
- Loads the quantized INT8 TensorFlow Lite model
- Loads the MNIST test dataset
- Evaluates the model's accuracy on the test data
- Prints the accuracy of the quantized model

## Installation Requirements

To run these scripts, you need:

1. Python 3.7 or later
2. TensorFlow 2.x
3. NumPy

**Note: You can install the required packages using pip**