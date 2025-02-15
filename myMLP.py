import numpy as np
from activations import ReLU, Sigmoid, Tanh, Linear

class myMLP:
    def __init__(self, widths, activations, initializer="xavier"):
        self.W = []
        self.b = []
        self.num_layers = len(widths)

        if len(widths) != len(activations) + 1:
            raise ValueError("Number of layer widths must be one more than number of activations")
        
        self.activations = []

        for a in activations:
            if a == "relu":
                self.activations.append(ReLU())
            elif a == "sigmoid":
                self.activations.append(Sigmoid())
            elif a == "tanh":
                self.activations.append(Tanh())
            elif a == "linear":
                self.activations.append(Linear())

        for i in range(self.num_layers-1):
            input_size = widths[i]
            output_size = widths[i+1]
            if initializer == "xavier":
                self.W.append(self.xavier_initialize(input_size, output_size))
            elif initializer == "he":
                self.W.append(self.he_initialize(input_size, output_size))
            self.b.append(np.zeros((1, output_size)))
    
    def xavier_initialize(self, input_size, output_size):
        std = np.sqrt(6. / (input_size + output_size))
        return np.random.uniform(-std, std, size=(input_size, output_size))
    
    def he_initialize(self, input_size, output_size):
        std = np.sqrt(2. / input_size)
        return np.random.normal(0, std, size=(input_size, output_size))
    
    def forward(self, X):
        y = [X]
        z = []
        for i in range(self.num_layers-1):
            z.append(np.dot(y[i], self.W[i]) + self.b[i])
            y.append(self.activations[i].activate(z[i]))
        return y, z
    
    def backward(self, y_true, y_pred, z, loss_fn):
        dW = [np.zeros_like(w) for w in self.W]
        db = [np.zeros_like(b) for b in self.b]
        dL = loss_fn.gradient(y_true, y_pred[-1])

        for i in reversed(range(self.num_layers-1)):
            dL = dL * self.activations[i].derivative(z[i])
            dW[i] = np.dot(y_pred[i].T, dL)
            db[i] = np.sum(dL, axis=0, keepdims=True)
            dL = np.dot(dL, self.W[i].T)
        
        return dW, db
        
    def train_MLP(self, X_train, y_train, X_val, y_val, num_epochs, loss_fn, optimizer):
        logs = {}
        logs['train_loss'] = []
        logs['test_loss'] = []
        for epoch in range(num_epochs):
            y_pred, z = self.forward(X_train)
            y_pred_val, _ = self.forward(X_val)

            dW, db = self.backward(y_train, y_pred, z, loss_fn)
            optimizer.step(self, dW, db)

            loss = loss_fn.loss(y_train, y_pred[-1])
            loss_val = loss_fn.loss(y_val, y_pred_val[-1])

            logs['train_loss'].append(loss)
            logs['test_loss'].append(loss_val)
            
            print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {loss:.4f}, Val Loss: {loss_val:.4f}")

        return logs
    
    def predict(self, X_test):
        y_pred, _ = self.forward(X_test)
        return y_pred[-1]
    
    def test_MLP(self, X_test, y_test, loss_fn):
        y_pred = self.predict(X_test)
        loss = loss_fn.loss(y_test, y_pred)
        return loss

