import numpy as np

class CrossEntropy:
    def __init__(self):
        pass

    def loss(self, y_true, y_pred):
        y_pred = np.clip(y_pred, 1e-12, 1 - 1e-12)
        return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
    
    def gradient(self, y_true, y_pred):
        y_pred = np.clip(y_pred, 1e-12, 1 - 1e-12)
        return -(y_true / y_pred) + (1 - y_true) / (1 - y_pred)
    
class L2:
    def __init__(self):
        pass

    def loss(self, y_true, y_pred):
        return np.mean((y_true - y_pred) ** 2)
    
    def gradient(self, y_true, y_pred):
        return -2 * (y_true - y_pred)