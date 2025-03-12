from tensorcore import Tensor
import numpy as np

class SGD:

    def __init__(self, state_dict: dict, lr: float):
        self.state_dict = state_dict
        self.lr = lr

    def zero_grad(self):

        for layer, params in self.state_dict.items():
            w, b = params
            w.zero_grad()
            b.zero_grad()

    def step(self):
        # print('hello')
        for layer, params in self.state_dict.items():
            w, b = params
            w.data = w.data - self.lr*w.grad

            # if b:
            #     b.data = b.data - self.lr*b.grad
            # print(f'layer: {layer} | weight: {w.data}')

class Adam:
    pass

