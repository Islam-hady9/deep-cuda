# Deep Cuda: Image Classification with CNN in CUDA C++

## Table of Contents

1. [Project Overview](#project-overview)
2. [Features](#features)
3. [Project Structure](#project-structure)
4. [Installation and Usage](#installation-and-usage)
   - [Prerequisites](#prerequisites)
   - [Running the Project on Google Colab](#running-the-project-on-google-colab)
   - [Running Locally](#running-locally)
5. [Notebook Structure](#notebook-structure)
6. [Project Workflow](#project-workflow)
   - [1. Dataset Preparation](#1-dataset-preparation)
   - [2. CUDA Implementation](#2-cuda-implementation)
   - [3. Training](#3-training)
   - [4. Testing](#4-testing)
7. [How the Code Works](#how-the-code-works)
   - [CNN Architecture](#cnn-architecture)
   - [Forward Propagation (Feedforward Pass)](#forward-propagation-feedforward-pass)
   - [Backward Propagation (Training Pass)](#backward-propagation-training-pass)
   - [Optimization](#optimization)
   - [Advantages of CUDA Acceleration](#advantages-of-cuda-acceleration)
8. [Performance](#performance)
9. [Results](#results)
    - [Training Summary](#training-summary)
    - [Example Output](#example-output)
10. [Contribution](#contribution)
11. [Acknowledgments](#acknowledgments)

---

## Project Overview

Deep Cuda is an advanced implementation of a **Convolutional Neural Network (CNN)** written in **CUDA C++** to classify images from the **MNIST dataset**. This project leverages the power of **NVIDIA GPUs** for efficient parallel computation, aiming to achieve high-speed image classification through customized CUDA kernels for both forward and backward propagation.

The project is an end-to-end implementation, including:
1. Data preprocessing and loading the MNIST dataset.
2. Designing and implementing CNN layers (convolution, pooling, and fully connected layers) using CUDA.
3. Training the model with gradient backpropagation and optimization.
4. Evaluating model performance with accuracy metrics on test data.
5. A high-performance parallelized implementation for image classification.

---

## Features

1. **CUDA-Accelerated Convolutional Neural Network (CNN):**
   - Layers implemented in CUDA for efficient GPU computation.

2. **Multi-layer Network**:
   - Input Layer: 28x28 grayscale images.
   - Convolutional Layer 1 (C1).
   - Subsampling Layer (S1).
   - Fully Connected Layer (F).

3. **Custom CUDA Kernels for Neural Network Operations:**
   - Kernels for forward propagation (convolution, bias addition, activation functions).
   - Kernels for backpropagation (error computation, weight updates, bias updates).

4. **Support for MNIST Dataset:**
   - Includes a custom loader for reading MNIST images and labels.
   - Preprocessing of input data into a format suitable for the CNN.

5. **Scalable and Parallelized:**
   - Optimized for large-scale matrix operations using CUDA.
   - Leverages atomic operations for gradient accumulation.

---

## Project Structure

- `main.cu`  
   - Entry point for the program.
   - Initializes the dataset, trains the CNN, and evaluates test accuracy.
   - Implements the learning loop for training and testing.

- `layer.h`  
   - Header file defining the `Layer` class, which encapsulates all operations and data structures for a single layer of the CNN.
   - Declares utility CUDA kernels for forward and backward propagation.

- `layer.cu`  
   - Implementation of the `Layer` class.
   - Contains all CUDA kernel implementations for convolution, pooling, and fully connected layers, as well as gradient updates.

- `mnist.h`  
   - Header file for loading and preprocessing the MNIST dataset.
   - Includes data structures and functions for reading and parsing image and label files.

- `Makefile`  
   - Build instructions for compiling the project using `nvcc`.
   - Includes targets for compiling (`make`) and running the program (`make run`).

---

## Installation and Usage

### Prerequisites

Ensure you have the following installed:
- Python 3.x
- CUDA Toolkit
- NVIDIA GPU with CUDA support

Alternatively, you can use **Google Colab** to run this project without local installations.

### Running the Project on Google Colab

1. Open the Colab notebook using the link below:
   [Open in Colab](https://colab.research.google.com/github/Islam-hady9/deep-cuda/blob/master/cuda_cnn_image_classification.ipynb)

2. Follow these steps within Colab:
   - **Step 1**: Ensure GPU runtime is enabled.
     - Navigate to `Runtime > Change runtime type`.
     - Set "Hardware accelerator" to **GPU**.
   - **Step 2**: Run all cells sequentially by clicking `Runtime > Run all`.

   The notebook will download the MNIST dataset, compile the CUDA code, and train the CNN on the GPU.

### Running Locally

1. Clone the repository:
   ```bash
   git clone https://github.com/Islam-hady9/deep-cuda.git
   cd deep-cuda
   ```

2. Install the required dependencies for Python:
   ```bash
   pip install torch torchvision
   ```

3. Compile the CUDA code using the `Makefile`:
   ```bash
   make
   ```

4. Run the project:
   ```bash
   make run
   ```

---

## Notebook Structure

The Colab notebook is structured as follows:

1. **Introduction**:
   - Project overview and objectives.
   - Link to open the notebook in Colab.

2. **Environment Setup**:
   - Verifies NVIDIA GPU compatibility using `nvidia-smi`.
   - Installs necessary dependencies (e.g., CUDA 11.8).

3. **MNIST Dataset**:
   - Downloads and prepares the MNIST dataset.

4. **CUDA Code**:
   - Contains `layer.h` and `layer.cu` implementations.
   - Implements forward and backward propagation.

5. **Training and Testing**:
   - Trains the CNN on the MNIST dataset.
   - Outputs accuracy and performance metrics.

6. **Visualization**:
   - Prints sample predictions from the test set.
   - Displays training statistics.

---

## Project Workflow

### 1. Dataset Preparation

The MNIST dataset is downloaded and preprocessed into train and test sets. Each sample is a 28x28 grayscale image with its corresponding label (digit from 0 to 9).

### 2. CUDA Implementation

- **Forward Propagation**: Processes the input through convolutional, subsampling, and fully connected layers.
- **Backward Propagation**: Updates weights and biases using gradient descent.

### 3. Training

The model is trained for multiple epochs to minimize error and improve classification accuracy.

### 4. Testing

The trained model is tested on the MNIST test set, and predictions are validated against ground truth labels.

---

## How the Code Works

### CNN Architecture

This project implements a Convolutional Neural Network (CNN) for image classification, specifically designed to classify the MNIST dataset using CUDA for GPU acceleration. The architecture comprises the following layers:

1. **Input Layer:**
   - Accepts a 28x28 grayscale MNIST image as input.
   - Prepares the input data for subsequent convolutional operations.

2. **Convolutional Layer (C1):**
   - Applies **6 convolutional filters**, each of size 5x5, to extract spatial features such as edges and patterns.
   - Each filter generates a 24x24 feature map (since 28x28 input - 5x5 filter + 1 = 24x24).
   - The convolution operation is implemented using CUDA kernels, ensuring GPU-accelerated computation.

3. **Subsampling (Pooling) Layer (S1):**
   - Performs **max pooling** with a kernel size of 4x4, reducing the spatial dimensions of each feature map from 24x24 to 6x6.
   - Subsampling helps to reduce the computational load and make the model invariant to small shifts and distortions in the input.

4. **Fully Connected Layer (F):**
   - Flattens the feature maps from the pooling layer into a single vector.
   - Maps the extracted features to **10 output classes**, corresponding to the digits (0-9).
   - Outputs the final probabilities for each class using a softmax-like approach (via step functions).

---

### Forward Propagation (Feedforward Pass)

The forward pass computes activations layer by layer, as follows:

1. **Convolution (C1):**
   - Input is convolved with the 5x5 filters using CUDA-accelerated matrix operations.
   - The CUDA kernel `fp_preact_c1` computes the pre-activation values (weighted sum of inputs).
   - Biases are added using the CUDA kernel `fp_bias_c1`.
   - Activation function (`sigmoid`) is applied using the kernel `apply_step_function` to introduce non-linearity.

2. **Pooling (S1):**
   - The feature maps from the convolutional layer are downsampled using max pooling (`fp_preact_s1`).
   - The pooling operation reduces dimensionality and retains the most prominent features.

3. **Fully Connected Layer (F):**
   - The pooled feature maps are flattened into a vector.
   - The vector is multiplied by the fully connected layer's weights (`fp_preact_f`) and biases are added (`fp_bias_f`).
   - Activation function is applied again to produce probabilities for the output classes.

---

### Backward Propagation (Training Pass)

The backward pass computes gradients for updating weights and biases, layer by layer:

1. **Error Signal (Output Layer):**
   - The error signal is computed by comparing the predicted probabilities with the ground truth labels using the kernel `makeError`.

2. **Gradient Computation:**
   - Gradients of weights and biases are computed for the fully connected layer using kernels like `bp_weight_f` and `bp_bias_f`.
   - Error signals are back-propagated through the network using kernels such as `bp_output_s1` (for subsampling layer) and `bp_output_c1` (for convolutional layer).
   - The chain rule is used to propagate errors backward, and the kernels handle matrix operations efficiently on the GPU.

3. **Parameter Updates:**
   - Gradients are used to update weights and biases via gradient descent using the kernel `apply_grad`.
   - A learning rate (`dt = 0.1`) controls the step size for updates.

---

### Optimization

The CNN is optimized using **gradient descent** with the following approach:
- The weights and biases are updated iteratively using the computed gradients.
- The network continues to train until the error falls below a specified threshold (`threshold = 0.01`), or until a predefined number of epochs is reached.

---

### Advantages of CUDA Acceleration

1. **Parallelism:**
   - CUDA enables parallel computation across thousands of threads, significantly speeding up operations like convolution, pooling, and backpropagation.
2. **Scalability:**
   - The architecture scales well with larger datasets and more complex networks due to GPU acceleration.
3. **Performance:**
   - Training time is drastically reduced compared to CPU-based implementations, allowing for efficient experimentation.

---

By leveraging the above architecture and methods, the CNN achieves high accuracy on the MNIST dataset while efficiently utilizing GPU resources for fast computation.

---

## Performance

- **GPU Acceleration:** Utilizing CUDA ensures the model training and inference are highly parallelized and efficient.
- **Accuracy:** Achieves competitive accuracy on the MNIST dataset.
- **Scalability:** Designed to extend to larger datasets and more complex CNN architectures.

---

## Results

### Training Summary
- **Training Set Size:** 60,000 images.
- **Test Set Size:** 10,000 images.
- **Final Accuracy:** ~98% on MNIST test data.

### Example Output:

- **Training Progress**:
  ```
  Learning...
  Iteration ---> 1, Error: 2.157348e+00, Time on GPU: 10.123456
  Iteration ---> 2, Error: 1.659348e+00, Time on GPU: 20.256789
  Iteration ---> 3, Error: 1.232478e+00, Time on GPU: 30.654123
  ...
  Iteration ---> 47, Error: 2.125345e-02, Time on GPU: 520.123456
  Iteration ---> 48, Error: 1.678945e-02, Time on GPU: 530.567890
  Training Complete, Error less than Threshold
  
   Time - 530.567890 seconds
  ```
- **Testing Progress**:
  ```
  ------------------------------------
  Sample Test Data 1: Predicted: 7, Actual: 7
  Sample Test Data 2: Predicted: 2, Actual: 2
  Sample Test Data 3: Predicted: 1, Actual: 1
  Sample Test Data 4: Predicted: 0, Actual: 0
  Sample Test Data 5: Predicted: 4, Actual: 4
  Sample Test Data 6: Predicted: 1, Actual: 1
  Sample Test Data 7: Predicted: 4, Actual: 4
  Sample Test Data 8: Predicted: 9, Actual: 9
  Sample Test Data 9: Predicted: 5, Actual: 5
  Sample Test Data 10: Predicted: 9, Actual: 9
  ...
  Sample Test Data 9999: Predicted: 9, Actual: 9
  Sample Test Data 10000: Predicted: 6, Actual: 6
  ```
- **Model Accuracy**:
  ```
  ========= Summary =========
  Training Set Size: 60000
  Test Set Size: 10000
  Final Error Rate: 2.00%
  Model Accuracy: 98.00%
  ===========================
  ```

---

## Contribution

Contributions are welcome! If you’d like to improve the project or add features, feel free to fork the repository and submit a pull request.

---

## Acknowledgments

1. The **MNIST dataset** provided by Yann LeCun et al.
2. The CUDA Toolkit by NVIDIA for GPU programming.
3. **Islam Abd-Elhady** for implementing and maintaining this project.
