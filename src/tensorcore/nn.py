import numpy as np
from tensorcore import Tensor

class Linear:
    def __init__(self, in_features, out_features, bias=True):
        self.weight = Tensor(np.random.uniform(low = 0, high=1, size=(in_features, out_features)), requires_grad=True)
        self.bias = Tensor(np.zeros(out_features)) if bias else None

    def forward(self, x):
        assert isinstance(x, Tensor), 'input must be of type Tensor'
        out = x.dot(self.weight)
        if self.bias:
            out = out + self.bias
        return out

    def __call__(self, x):
        return self.forward(x)

# loss function implementation

def mse(y, y_pred):
    assert isinstance(y, Tensor) and isinstance(y_pred, Tensor), "mse() expects two Tesor objects"
    loss = np.mean((y.data - y_pred.data)**2)
    out = Tensor(loss, (y, y_pred), _op='mse', requires_grad=True)

    def _backward():
        if out.requires_grad:
            # y.grad += -2 * (y_pred.data - y.data) / len(y.data) * out.grad
            y_pred.grad += 2 * (y_pred.data - y.data) / len(y.data) * out.grad
    out._backward = _backward
    return out

def binary_cross_entropy(y_true, y_pred, epsilon=1e-12):
    assert isinstance(y_pred, Tensor) and isinstance(y_true, Tensor), "binary_cross_entropy() expects two Tensor objects"
    y_pred_clipped = np.clip(y_pred.data, epsilon, 1 - epsilon)
    loss = -np.mean(y_true.data * np.log(y_pred_clipped) + (1 - y_true.data) * np.log(1 - y_pred_clipped))
    out = Tensor(loss, (y_true, y_pred), _op='binary_cross_entropy', requires_grad=True)

    def _backward():
        if out.requires_grad:
            y_pred.grad += (y_pred.data - y_true.data) / len(y_true.data) * out.grad
    out._backward = _backward
    return out

def categorical_cross_entropy(y_pred, y_true):
    assert isinstance(y_pred, Tensor) and isinstance(y_true, Tensor), "categorical_cross_entropy() expects two Tesor objects"
    loss = -np.sum(y_true.data * np.log(y_pred.data)) / len(y_true.data)
    out = Tensor(loss, (y_pred, y_true), _op='categorical_cross_entropy', requires_grad=True)

    def _backward():
        if out.requires_grad:
            y_pred.grad += -y_true.data / y_pred.data / len(y_true.data) * out.grad
    out._backward = _backward
    return out

# activation functions with automatic differentiation support
def sigmoid(x: Tensor):
    assert isinstance(x, Tensor), 'input must be of type Tensor'
    out = Tensor(1 / (1 + np.exp(-x.data)), (x,), requires_grad=True) 

    def _backward():
        if out.requires_grad:
            x.grad += out.data * (1 - out.data) * out.grad 
    out._backward = _backward
    return out