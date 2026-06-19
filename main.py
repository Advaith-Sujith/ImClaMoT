from tqdm import trange
import os
from PIL import Image
import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split


def img_processor(img_path):
    img = Image.open(img_path)
    resized = img.resize((64, 64)).convert("L")
    array = np.asarray(resized, dtype=np.float32)/255.0
    return array.flatten()

def process_db(seed,path):
    classes = sorted([
    dir for dir in os.listdir(path)
    if os.path.isdir(os.path.join(path, dir))
    ])
    X = []
    y = []
    for ind,dir in enumerate(classes):
        class_path = os.path.join(path, dir)

        for img_name in os.listdir(class_path):
            img_path = os.path.join(class_path, img_name)

            if os.path.isfile(img_path):
                X.append(img_processor(img_path))
                y.append(ind)
    X = np.array(X)
    y = np.array(y)

    rng = np.random.default_rng(seed)
    indices = rng.permutation(len(X))

    return X[indices], y[indices]

def pre_proc(X, y):
    xtrain, xtest, ytrain, ytest = train_test_split(X, y, test_size=0.3, random_state=42)
    return xtrain, xtest, ytrain, ytest

def init_param(features, classes, layers):

    weights = []
    biases = []

    root_neurons = features
    for i in range(len(layers)):
        weights.append(np.random.randn(root_neurons, layers[i]) * np.sqrt(2/root_neurons))
        biases.append(np.zeros((1, layers[i])))
        root_neurons = layers[i]
    
    weights.append(np.random.randn(root_neurons, classes) * np.sqrt(2/root_neurons))
    biases.append(np.zeros((1, classes)))

    return weights, biases

def ReLU(z):

    return np.maximum(0,z)

def SoftMax(z):

    z = z - np.max(z, axis=1, keepdims=True)
    return np.exp(z)/np.sum(np.exp(z), axis=1, keepdims=True)

def cost(y, a, classes):

    y = onehotencode(y, classes)
    m = y.shape[0]
    loss = y*np.log(a+1e-8)
    cost = -np.sum(loss)/m

    return loss, cost

def accuracy(y_true, y_pred):

    preds = np.argmax(y_pred, axis=1)
    return np.mean(preds == y_true)

def forward(x, weights, biases):

    z = []
    a = []
    z.append(np.dot(x, weights[0])+biases[0])
    a.append(ReLU(z[0]))
    for i in range(1, len(weights)-1):
        z.append(np.dot(a[i-1], weights[i])+biases[i])
        a.append(ReLU(z[i]))
    z.append(np.dot(a[-1], weights[-1])+biases[-1])
    a.append(SoftMax(z[-1]))

    return z, a

def relu_der(z):

    return (z>0).astype(float)

def onehotencode(y, classes):

    return np.eye(classes)[y]

def backprop(x, y, a, z, classes, weights):

    y = onehotencode(y, classes)
    m = y.shape[0]
    dz = a[-1] - y
    dw = [None]*len(weights)
    db = [None]*len(weights)

    dw[-1] = (1/m) * np.dot(a[-2].T, dz)
    db[-1] = (1/m) * np.sum(dz, axis=0, keepdims=True)

    for i in range(len(weights)-2, -1, -1):
        dz = np.dot(dz,weights[i+1].T) * relu_der(z[i])

        if i==0:
            dw[i] = (1/m) * np.dot(x.T, dz)
        else:
            dw[i] = (1/m) * np.dot(a[i-1].T, dz)
        
        db[i] = (1/m) * np.sum(dz, axis=0, keepdims=True)
    
    return dw, db

def improv(weights, biases, dw, db, lr):

    for i in range(len(weights)):
        weights[i] -= lr*dw[i]
        biases[i] -= lr*db[i]
    
    return weights, biases

def fit(X, y, features, classes, epochs, lr, layers=[64, 32]):

    xtrain, xtest, ytrain, ytest = pre_proc(X, y)
    pbar = trange(epochs, desc="Training")
    weights, biases = init_param(features, classes, layers)

    for epoch in pbar:

        indices = np.random.permutation(len(xtrain))
        xtrain = xtrain[indices]
        ytrain = ytrain[indices]

        best_acc = 0
        best_w = None
        best_b = None


        batch_size = 64
        for start in range(0, len(xtrain), batch_size):
            end = start + batch_size

            x = xtrain[start:end]
            y = ytrain[start:end]
            z, a = forward(x, weights, biases)

            dw, db = backprop(x, y, a, z, classes, weights)

            weights, biases = improv(weights, biases, dw, db, lr)

            if test_acc > best_acc:
                best_acc = test_acc
                best_w = [w.copy() for w in weights]
                best_b = [b.copy() for b in biases]

        if epoch % 10 == 0:
            _, train_a = forward(xtrain, best_w, best_b)
            train_acc = accuracy(ytrain, train_a[-1])
            _, train_cost = cost(ytrain, train_a[-1], classes)
            _, test_a = forward(xtest, best_w, best_b)
            test_acc = accuracy(ytest, test_a[-1])

            pbar.set_postfix(
                train_acc=f"{train_acc:.4f}",
                test_acc=f"{test_acc:.4f}",
                loss=f"{train_cost:.4f}")

    _, test_a = forward(xtest, best_w, best_b)

    test_loss, test_cost = cost(ytest, test_a[-1], classes)
    test_accuracy = accuracy(ytest, test_a[-1])

    print("Test cost: ", test_cost)
    print("Test accuracy: ", test_accuracy)
    
    return best_w, best_b, test_loss, test_cost, test_accuracy

def model_save(weights, biases, path="model.npz"):
    data = {}

    for i, w in enumerate(weights):
        data[f"W{i}"] = w

    for i, b in enumerate(biases):
        data[f"b{i}"] = b

    np.savez(path, **data)

def predict(img_path, weights, biases):
    x = img_processor(img_path)
    x = x.reshape(1, -1)

    _, a = forward(x, weights, biases)

    probs = a[-1]
    pred = np.argmax(probs, axis=1)[0]

    return pred, probs[0]

