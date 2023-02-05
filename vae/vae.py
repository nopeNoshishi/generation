import numpy as np

class Layer():
    def __init__(self):
        pass

    def __call__(self, input):
        y = self.forward(input)
        return y
    
    def forward(self, input):
        raise NotImplementedError()

class Dence(Layer):
    def __init__(self, in_n: int, out_n: int):
        self.in_n = in_n
        self.out_n = out_n

        self.W = np.random.randn(out_n, in_n) * np.sqrt(in_n)
        self.b = np.random.randn(out_n) * np.sqrt(in_n)

    def forward(self, x):
        out = np.dot(self.W, x) * self.b

        self.x = x
        return out
    
class Sigmoid(Layer):
    def __init__(self):
        self.out = None

    def forward(self, x):
        out = 1 / (1 + np.exp(-x))
        self.out = out

        return out
    

# def leru(x):
#     return np.where(x > 0.0, x, 0.0)

# def leaky_relu(x, alpha=0.01):
#   return np.where(x >= 0.0, x, alpha * x)


input = np.random.randn(20,10)
print(input)

dence_layer = Dence(20, 10)
sigmoid_layer = Sigmoid()

y = dence_layer(input)
print(y)

y = sigmoid_layer(y)
print(y)
