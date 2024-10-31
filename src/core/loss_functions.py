from copy import deepcopy
from typing import Callable, Tuple

import numpy as np

LossFunc = Callable[[np.ndarray, np.ndarray], Tuple[int, np.ndarray]]


def apply_svm(x, y):
    """
    Computes the loss and gradient using for multiclass SVM classification.

    Inputs:
    - x: Input data, of shape (N, C) where x[i, j] is the score for the jth
      class for the ith input.
    - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
      0 <= y[i] < C

    Returns a tuple of:
    - loss: Scalar giving the loss
    - dx: Gradient of the loss with respect to x
    """
    N = x.shape[0]
    delta = 1.0

    correct_class_scores = x[np.arange(N), y].reshape(N, 1)
    margins = np.maximum(0, x - correct_class_scores + delta)
    margins[np.arange(N), y] = 0

    loss = np.sum(margins) / N

    # gradient calculation
    dx = np.zeros_like(x)
    dx[margins > 0] = 1
    incorrect_counts = np.sum(margins > 0, axis=1)  # margin condition
    dx[np.arange(N), y] -= incorrect_counts
    dx /= N

    return loss, dx


def apply_softmax(X, y):
    """
    Computes the loss and gradient for softmax classification.

    Inputs:
    - X: Input data, of shape (N, C) where x[i, j] is the score for the jth
      class for the ith input.
    - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
      0 <= y[i] < C

    Returns a tuple of:
    - loss: Scalar giving the loss
    - dx: Gradient of the loss with respect to x
    """
    num_train = X.shape[0]
    scores = X

    exp_scores = np.exp(scores)
    probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
    dx = deepcopy(probs)
    correct_class_probs = probs[np.arange(num_train), y]

    loss = np.sum(-np.log(correct_class_probs))
    loss /= num_train

    # gradient calculation
    dx[np.arange(num_train), y] -= 1  # probs of correct class - 1
    dx /= num_train

    return loss, dx
