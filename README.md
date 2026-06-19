# ImClaMoT

Image Classification Model Trainer (ImClaMoT) is a lightweight image classification framework built entirely with NumPy. The project implements a fully connected feedforward neural network from scratch without relying on deep learning libraries such as TensorFlow or PyTorch.

The model is designed to classify grayscale images by training on user-provided image datasets organized into class folders.

## Features

* Neural network implementation from scratch using NumPy
* Custom forward propagation and backpropagation
* ReLU activation function
* Softmax output layer
* Cross-entropy loss
* He weight initialization
* Mini-batch gradient descent
* Model saving using `.npz` files
* Image preprocessing pipeline
* Multi-class image classification support
* Progress monitoring with tqdm

---

## Project Structure

```
ImClaMoT/
│
├── dataset/
│   ├── class_1/
│   │   ├── image1.jpg
│   │   ├── image2.jpg
│   │   └── ...
│   │
│   ├── class_2/
│   │   ├── image1.jpg
│   │   ├── image2.jpg
│   │   └── ...
│   │
│   └── ...
│
├── model.npz
├── imclamot.py
├── requirements.txt
└── README.md
```

### Dataset Format

Images must be stored inside folders corresponding to their class labels.

Example:

```
dataset/
├── cats/
│   ├── cat1.jpg
│   ├── cat2.jpg
│
├── dogs/
│   ├── dog1.jpg
│   ├── dog2.jpg
│
└── birds/
    ├── bird1.jpg
    └── bird2.jpg
```

Each folder name represents a classification category.

---

## Image Preprocessing

Each image undergoes the following preprocessing steps:

1. Load image using Pillow
2. Resize to 64 × 64 pixels
3. Convert to grayscale
4. Normalize pixel values to range [0,1]
5. Flatten into a vector of length 4096

```
64 × 64 = 4096 input features
```

---

## Neural Network Architecture

Default architecture:

```
Input Layer      : 4096 neurons
Hidden Layer 1   : 64 neurons
Hidden Layer 2   : 32 neurons
Output Layer     : Number of classes
```

Activation functions:

* Hidden Layers → ReLU
* Output Layer → Softmax

Loss Function:

* Cross Entropy Loss

Weight Initialization:

* He Initialization

---

## Training Pipeline

### 1. Load Dataset

```python
X, y = process_db(seed=42, path="dataset")
```

### 2. Train Model

```python
weights, biases, loss, cost, accuracy = fit(
    X,
    y,
    features=4096,
    classes=len(np.unique(y)),
    epochs=100,
    lr=0.001
)
```

### 3. Save Model

```python
model_save(weights, biases)
```

---

## Prediction

Predict a single image:

```python
predicted_class, probabilities = predict(
    "sample.jpg",
    weights,
    biases
)
```

Returns:

```python
predicted_class
```

and

```python
[class probabilities]
```

---

## Training Details

Dataset Split:

```
70% Training
30% Testing
```

Mini-batch Size:

```
64
```

Optimizer:

```
Mini-batch Gradient Descent
```

Evaluation Metrics:

* Classification Accuracy
* Cross Entropy Loss

---

## Dependencies

Install required packages:

```bash
pip install numpy pillow scikit-learn tqdm
```

Required libraries:

* NumPy
* Pillow
* Scikit-learn
* tqdm

---

## Example Usage

```python
X, y = process_db(
    seed=42,
    path="dataset"
)

weights, biases, _, _, acc = fit(
    X,
    y,
    features=4096,
    classes=len(np.unique(y)),
    epochs=100,
    lr=0.001
)

model_save(weights, biases)

prediction, probs = predict(
    "test.jpg",
    weights,
    biases
)
```

---

## Limitations

* Fully connected architecture only
* No convolutional layers
* Grayscale images only
* Fixed image resolution of 64×64
* CPU training only
* No data augmentation
* No learning rate scheduling
* No regularization techniques implemented

---

## Project Status

Archived — Development Complete

This project was created as an educational implementation of a neural network-based image classifier using only NumPy and basic scientific computing libraries. No further development or feature additions are planned.

---

## License

This project is provided for educational and research purposes. Feel free to use, modify, and extend the code for personal or academic work.
