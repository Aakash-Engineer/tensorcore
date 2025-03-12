import numpy as np

import numpy as np

class Tensor:
    def __init__(self, data, _children=(), _op='', label='', requires_grad=False):
        self.data = np.array(data, dtype=np.float32)
        self.requires_grad = requires_grad
        self.grad = np.zeros_like(self.data) if requires_grad else None
        self._prev = set(_children)
        self._backward = lambda: None
        self.label = label
        self._op = _op
        self.shape = self.data.shape
        self.dim = self.data.ndim

    def __add__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)

        # Perform element-wise addition with broadcasting
        out = Tensor(self.data + other.data, (self, other), '+', requires_grad=self.requires_grad or other.requires_grad)

        if out.requires_grad:
            def _backward():
                if self.requires_grad:
                    grad_self = out.grad
                    if self.shape != out.shape:
                        grad_self = self._reduce_broadcasted_grad(grad_self, self.shape)
                    self.grad += grad_self

                if other.requires_grad:
                    grad_other = out.grad
                    if other.shape != out.shape:
                        grad_other = other._reduce_broadcasted_grad(grad_other, other.shape)
                    other.grad += grad_other

            out._backward = _backward
        return out

    def _reduce_broadcasted_grad(self, grad, original_shape):
        """
        Reduces the gradient to match the original shape before broadcasting.
        """
        if grad.shape == original_shape:
            return grad  # No need to reduce

        # Sum along the dimensions that were broadcasted
        axis = tuple(i for i, (g_dim, orig_dim) in enumerate(zip(grad.shape, original_shape)) if orig_dim == 1)
        grad = grad.sum(axis=axis, keepdims=True)  # Reduce along broadcasted dims

        # Remove unnecessary dimensions
        grad = grad.reshape(original_shape)

        return grad

    def __mul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data * other.data, (self, other), '*', requires_grad=self.requires_grad or other.requires_grad)

        if out.requires_grad:
            def _backward():
                if self.requires_grad:
                    self.grad += other.data * out.grad
                if other.requires_grad:
                    other.grad += self.data * out.grad

            out._backward = _backward
        return out

    def __pow__(self, exponent):
        assert isinstance(exponent, (int, float)), 'Only int and float allowed as exponent'
        out = Tensor(self.data ** exponent, (self,), '**', requires_grad=self.requires_grad)

        if out.requires_grad:
            def _backward():
                if self.requires_grad:
                    self.grad += (exponent * (self.data ** (exponent - 1))) * out.grad

            out._backward = _backward
        return out

    def backward(self):
        if self.requires_grad:
            top = []
            visited = set()

            def build_topo(v):
                if v not in visited:
                    visited.add(v)
                    for child in v._prev:
                        build_topo(child)
                    top.append(v)

            build_topo(self)
            self.grad = np.ones_like(self.data)  # Initialize gradient as ones for the output node
            for node in reversed(top):
                node._backward()

    def zero_grad(self):
        if self.requires_grad:
            self.grad = np.zeros_like(self.data)

    def sum(self):
        out = Tensor(np.sum(self.data), (self,), _op='sum', requires_grad=self.requires_grad)
        def _backward():
            if out.requires_grad:
                self.grad += (1*out.grad)
        out._backward = _backward
        return out

    def dot(self, other):
        out = Tensor(np.dot(self.data, other.data), (self, other), _op='dot', requires_grad=self.requires_grad or other.requires_grad)

        def _backward():
            if out.requires_grad:
                if self.requires_grad:
                    self.grad += np.dot(out.grad, other.data.T)
                if other.requires_grad:
                    other.grad += np.dot(self.data.T, out.grad)

        out._backward = _backward
        return out

    def sigmoid(self):
        out = Tensor((1/ (1+ np.exp(-self.data))), (self,), _op='sigmoid', requires_grad=self.requires_grad)

        def _backward():
            if out.requires_grad:
                self.grad += (1-out.data) * out.grad

        out._backward = _backward
        return out

    def softmax(self):
        out = Tensor(np.exp(self.data) / (np.exp(self.data)).sum(), (self,), _op='siftmax', requires_grad=self.requires_grad)

        def _backward():
            if out.requires_grad:
                if out.data.ndim == 1:  # Single vector case
                    jacobian = np.diag(out.data) - np.outer(out.data, out.data)
                    self.grad += np.dot(jacobian, out.grad.reshape(-1, 1)).flatten()  # Ensure correct shape
                elif out.data.ndim == 2:  # Batch case
                    batch_size, n = out.data.shape
                    self.grad += np.einsum('bij,bj->bi', np.diag(out.data) - np.einsum('bi,bj->bij', out.data, out.data), out.grad)

        out._backward = _backward
        return out

    def __getitem__(self, index):
    # Capture the index during the forward pass
        out_data = self.data[index]
        out = Tensor(out_data, (self,), _op='indexing', requires_grad=self.requires_grad)

        if out.requires_grad:
            def _backward():
                if self.requires_grad:
                    # Propagate gradients back to the original positions
                    grad = np.zeros_like(self.data)
                    grad[index] += out.grad  # Add gradients to the indexed positions
                    self.grad += grad  # Accumulate gradients
            out._backward = _backward

        return out

    def mse(self, y):
        assert isinstance(y, Tensor), "mse() expects a Tesor object"
        loss = np.mean((y.data - self.data)**2)
        out = Tensor(loss, (self, y), _op='mse', requires_grad=self.requires_grad)

        def _backward():
            if out.requires_grad:
                self.grad += -2 * (y.data - self.data.reshape(y.data.shape)) / len(y.data) * out.grad
        out._backward = _backward
        return out

    def __neg__(self):
        return self * -1

    def __radd__(self, other):
        return self + other

    def __sub__(self, other):
        return self + (-other)

    def __rsub__(self, other):
        return other + (-self)

    def __rmul__(self, other):
        return self * other

    def __truediv__(self, other):
        return self * other**-1

    def __rtruediv__(self, other):
        return other * self**-1

    def __repr__(self):
        return f"Tensor({self.data})"

    @classmethod
    def rand(cls, *shape):
        return cls(np.random.rand(*shape), requires_grad=False)


class Sequential:
    def __init__(self, layers=()):
        self.layer_list = layers
        self.state_dict = {f"layer_{i}": (self.layer_list[i].weight, self.layer_list[i].bias) for i in range(len(self.layer_list))}

    def forward(self, x):
        for l in self.layer_list:
            x = l(x)
        return x

    def __call__(self, x):
        return self.forward(x)

    def parameters(self):
        return self.state_dict
    
    def __repr__(self):
        # return a good visual representation of the model in a string
        # return f"{self.__class__.__name__}({[layer for layer in self.layer_list]})"
        return 'hello'

