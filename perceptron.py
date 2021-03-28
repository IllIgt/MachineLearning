import numpy as np
from typing import Any

class Perceptron:
    """
        eta - learning rate
        n_iter - iteration count
        random_state - initial value of the random number generator
        w - weights
        errors - classification errors
    """

    def __init__(self, eta: float = 0.01, n_iter: int = 50, random_state: int = 1):
        self._eta = eta
        self._n_iter = n_iter
        self._random_state = random_state
        self._w = []
        self._errors = []

    def fit(self, X, y):
        random_gen = np.random.RandomState(self._random_state)
        self._w = random_gen.normal(loc=0.0, scale=0.01, size=1+X.shape[1])

        for _ in range(self._n_iter):
            errors = 0
            for xi, target in zip(X, y):
                update = self._eta * (target - self.predict(xi))
                self._w[1:] += update * xi
                self._w[0] += update
                errors += int(update != 0.0)
            self._errors.append(errors)
        return self

    def net_input(self, X: Any):
        return np.dot(X, self._w[1:]) + self._w[0]

    def predict(self, X: Any):
        return np.where(self.net_input(X) >= 0.0, 1, -1)

