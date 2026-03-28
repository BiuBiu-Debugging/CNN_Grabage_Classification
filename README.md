CNN for Binary Image Classification

This repository contains a complete PyTorch pipeline for training and evaluating a custom Convolutional Neural Network (CNN) for binary image classification. The project includes data augmentation, model building, training loops, validation, and performance visualization.

Project Structure

`Models.py`: Contains the PyTorch neural network architecture (`CNN_1`). The model consists of 6 convolutional layers, max-pooling layers, and fully connected layers, ending with a Sigmoid activation function suitable for binary classification.
`augment_and_train.ipynb`: A Jupyter Notebook that sets up the data loaders with image augmentations (resizing, center cropping, random horizontal flips, and random rotations), initializes the model, and runs the basic training loop using Binary Cross Entropy Loss (`BCELoss`) and the Adam optimizer.
`TEST.ipynb`: An extended version of the training notebook. It tracks both training and validation metrics (loss and accuracy) across all epochs and uses `matplotlib` to plot these metrics, allowing you to easily visualize model convergence and check for overfitting.

Requirements

To run this project, you will need Python 3 and the following libraries:

 `torch`
 `torchvision`
 `matplotlib`
 `jupyter` (to run the `.ipynb` notebooks)

You can install the dependencies using pip:
```bash
pip install torch torchvision matplotlib jupyter

Model Architecture Details

The `CNN_1` model is a custom Convolutional Neural Network designed for binary classification. It processes 3-channel RGB images (resized and center-cropped to 224x224) through three main convolutional blocks followed by a fully connected classifier.

1. Convolutional Block 1
Conv2D Layer 1: 3 input channels ➔ 32 output channels (Kernel size: 3x3, Padding: 0) + ReLU activation
Conv2D Layer 2: 32 input channels ➔ 32 output channels (Kernel size: 3x3, Padding: 0) + ReLU activation
MaxPooling2D: Kernel size 2x2, Stride 2

2. Convolutional Block 2
Conv2D Layer 3: 32 input channels ➔ 64 output channels (Kernel size: 3x3, Padding: 0) + ReLU activation
Conv2D Layer 4: 64 input channels ➔ 64 output channels (Kernel size: 3x3, Padding: 0) + ReLU activation
MaxPooling2D: Kernel size 2x2, Stride 2

3. Convolutional Block 3
Conv2D Layer 5: 64 input channels ➔ 128 output channels (Kernel size: 3x3, Padding: 0) + ReLU activation
Conv2D Layer 6: 128 input channels ➔ 128 output channels (Kernel size: 3x3, Padding: 0) + ReLU activation
MaxPooling2D: Kernel size 2x2, Stride 2

4. Fully Connected (Dense) Classifier
Flatten: Flattens the 2D feature maps into a 1D vector (size: 73,728).
Linear Layer 1: 73,728 input features ➔ 512 output features + ReLU activation
Dropout: 50% probability (p=0.5) to prevent overfitting.
Linear Layer 2: 512 input features ➔ 128 output features + ReLU activation
Linear Layer 3 (Output): 128 input features ➔ 1 output feature + Sigmoid activation (outputs a probability between 0 and 1 for binary classification).
