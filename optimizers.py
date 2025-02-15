import numpy as np

class GradientDescent:
    def __init__(self, learning_rate=0.01):
        self.learning_rate = learning_rate

    def step(self, model, dW, db):
        for i in range(len(model.W)):
            model.W[i] -= self.learning_rate * dW[i]
            model.b[i] -= self.learning_rate * db[i]
    
class Adam:
    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.momentum = None
        self.velocity = None
        self.time = 0

    def step(self, model, dW, db):
        self.time += 1
        if self.momentum is None:
            self.momentum = {}
            self.velocity = {}
            self.momentum['W'] = [np.zeros_like(w) for w in model.W]
            self.momentum['b'] = [np.zeros_like(b) for b in model.b]
            self.velocity['W'] = [np.zeros_like(w) for w in model.W]
            self.velocity['b'] = [np.zeros_like(b) for b in model.b]

        for i in range(len(model.W)):
            self.momentum['W'][i] = self.beta1 * self.momentum['W'][i] + (1 - self.beta1) * dW[i]
            self.momentum['b'][i] = self.beta1 * self.momentum['b'][i] + (1 - self.beta1) * db[i]
            self.velocity['W'][i] = self.beta2 * self.velocity['W'][i] + (1 - self.beta2) * (dW[i] ** 2)
            self.velocity['b'][i] = self.beta2 * self.velocity['b'][i] + (1 - self.beta2) * (db[i] ** 2)

            m_hat = self.momentum['W'][i] / (1 - self.beta1 ** self.time)
            v_hat = self.velocity['W'][i] / (1 - self.beta2 ** self.time)

            model.W[i] -= self.learning_rate * m_hat / (np.sqrt(v_hat) + self.epsilon)

            m_hat = self.momentum['b'][i] / (1 - self.beta1 ** self.time)
            v_hat = self.velocity['b'][i] / (1 - self.beta2 ** self.time)

            model.b[i] -= self.learning_rate * m_hat / (np.sqrt(v_hat) + self.epsilon)
class Momentum:
    def __init__(self, learning_rate=0.01, momentum=0.9):
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.velocity = None

    def step(self, model, dW, db):
        if self.velocity is None:
            self.velocity = {}
            self.velocity['W'] = [np.zeros_like(w) for w in model.W]
            self.velocity['b'] = [np.zeros_like(b) for b in model.b]

        for i in range(len(model.W)):
            self.velocity['W'][i] = self.momentum * self.velocity['W'][i] - self.learning_rate * dW[i]
            self.velocity['b'][i] = self.momentum * self.velocity['b'][i] - self.learning_rate * db[i]

            model.W[i] += self.velocity['W'][i]
            model.b[i] += self.velocity['b'][i]
