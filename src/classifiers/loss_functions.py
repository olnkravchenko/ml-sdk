from typing import Callable, Tuple

import numpy as np

LossFunc = Callable[[np.ndarray, np.ndarray, np.ndarray], int]


def softmax_loss_vectorized(W, X, y, reg) -> Tuple[float, np.array]:
    """
    Softmax loss function, vectorized version

    """
    # initialize the loss and gradient to zero
    loss = 0.0
    dW = np.zeros_like(W)

    num_train = X.shape[0]

    scores = X.dot(W)
    scores -= np.max(scores, axis=1).reshape(num_train, 1)

    exp_scores = np.exp(scores)
    probs = exp_scores / np.sum(exp_scores, axis=1).reshape(num_train, 1)
    correct_class_probs = probs[np.arange(num_train), y]

    loss = np.sum(-np.log(correct_class_probs))
    # gradient calculation
    correct_class_ind = np.zeros_like(probs)
    correct_class_ind[np.arange(num_train), y] = 1

    dW = X.T.dot(probs - correct_class_ind)

    loss /= num_train
    dW /= num_train

    loss += 0.5 * reg * np.sum(W * W)
    dW += reg * W

    return loss, dW
