import numpy as np
from typing import Optional, Tuple, List

class LogisticRegression:

    def __init__(self, num_features: int, 
                 eps: Optional[float] = 1e-12) -> None:
        self.weight = np.random.randn(num_features, 1) * 0.01
        self.bias = 0.0
        self.eps = eps
    
    def forward(self, inputs: np.ndarray) -> np.ndarray:
        assert inputs.shape[1] == self.weight.shape[0], \
            "size mismatch between inputs and weight"
        assert inputs.ndim == 2, \
            "inputs must be 2D"
        
        z = np.dot(inputs, self.weight) + self.bias
        a = sigmoid(z)
        return a

    def backward(self, inputs: np.ndarray, 
                 outputs: np.ndarray, 
                 labels: np.ndarray) -> Tuple[np.ndarray]:
        assert (inputs.shape[0] == outputs.shape[0] == labels.shape[0]), \
            "size mismatch between arrays"
        assert inputs.ndim == labels.ndim == 2, \
            "arrays must be 2D"
        assert labels.shape[1] == 1, \
            "labels array must be column array"
        
        m = len(labels)
        dz = outputs - labels
        dw = np.dot(inputs.T, dz) / m
        db = np.sum(dz, axis=0) / m
        gradients = dw, db
        return gradients

    def optimize(self, gradients: Tuple[np.ndarray], 
                 learning_rate: float) -> None:
        dw, db = gradients
        assert dw.shape[0] == self.weight.shape[0], \
            "size mismatch with dw and weight"
        assert dw.ndim == 2, \
            "weight gradient must be 2D"
        assert db.shape[0] == 1, \
            "bias gradient is not a scalar"
        
        self.weight -= learning_rate * dw
        self.bias -= learning_rate * db

    def predict(self, outputs: np.ndarray) -> np.ndarray:
        predictions = np.where(outputs > 0.5, 1, 0)
        return predictions

    def train(self, inputs: np.ndarray, 
              labels: np.ndarray, 
              iterations: int, 
              learning_rate: Optional[float] = 1e-3, 
              verbose: Optional[bool] = False) -> Tuple[float, List[float]]:
        assert (inputs.shape[0] == labels.shape[0]), \
            "size mismatch between arrays"
        assert inputs.ndim == labels.ndim == 2, \
            "arrays must be 2D"
        assert labels.shape[1] == 1, \
            "labels array must be column array"
        
        if verbose:
            print("training started")
        losses = []

        for _ in range(iterations):
            outputs = self.forward(inputs)
            loss = calculate_loss(outputs, labels, eps=self.eps)
            cost = np.mean(loss, axis=0)
            losses.append(cost)
            gradients = self.backward(inputs, outputs, labels)
            self.optimize(gradients, learning_rate)

        average_loss = np.mean(losses)
        if verbose:
            print(f"training complete with average loss: {average_loss:.4f}")
        return average_loss, losses
    
    def test(self, inputs: np.ndarray, 
             labels: np.ndarray, 
             verbose: Optional[bool] = False) -> Tuple[float]:
        assert (inputs.shape[0] == labels.shape[0]), \
            "size mismatch between arrays"
        assert inputs.ndim == labels.ndim == 2, \
            "arrays must be 2D"
        assert labels.shape[1] == 1, \
            "labels array must be column array"
        
        if verbose:
            print("testing started")
        m = len(labels)
        outputs = self.forward(inputs)
        predictions = self.predict(outputs)
        loss = np.sum(calculate_loss(outputs, labels), axis=0).item() / m
        correct = np.sum(predictions == labels)
        accuracy = correct / m

        if verbose:
            print(f"testing complete with average loss: {loss:.4f} and accuracy: {accuracy:.4f}")
        return loss, accuracy

def sigmoid(z: np.ndarray) -> np.ndarray:
    z = np.clip(z, -10, 10)
    a = 1 / (1 + np.exp(-z))
    return a

def calculate_loss(a: np.ndarray, y: np.ndarray, 
                   eps: Optional[float] = 1e-12) -> np.ndarray:
    a = np.clip(a, eps, 1 - eps)
    loss = -1 * (y * np.log(a) + (1 - y) * np.log(1 - a))
    return loss

def normalize(x: np.ndarray) -> np.ndarray:
    x_norm = np.linalg.norm(x, ord=2)
    x_normalized = x / x_norm
    return x_normalized

def flatten(x: np.ndarray) -> np.ndarray:
    x_flattened = x.reshape(x.shape[0], -1)
    return x_flattened