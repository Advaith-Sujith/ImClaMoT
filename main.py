import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split

def init_param(features, classes, layers=[64, 32]):

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

def fit(x_train, y_train, features, classes, layers, epochs):

    weights, biases = init_param(features, classes, layers)

    for epoch in range(epochs):

        x = x_train
        y = y_train

        z, a = forward(x, weights, biases)

        dw, db = backprop(x, y, a, z, classes, weights)

        weights, biases = improv(weights, biases, dw, db, 0.07)
    
    return weights, biases