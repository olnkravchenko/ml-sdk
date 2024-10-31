from __future__ import print_function

from builtins import object, range
from copy import deepcopy
from typing import Optional

import numpy as np
from loss_functions import LossFunc, softmax_loss_vectorized
from preprocessing import augmentation


class LinearClassifier(object):

    def __init__(
        self,
        weights: Optional[np.array] = None,
        loss_function: LossFunc = softmax_loss_vectorized,
    ):
        self.W: np.array = weights
        self.loss = loss_function

    def train(
        self,
        X: np.array,
        y: np.array,
        learning_rate=1e-3,
        reg=1e-5,
        num_iters=100,
        batch_size=200,
        verbose=False,
        is_apply_augmentation=True,
    ):
        """
        Train this linear classifier using stochastic gradient descent
        for N training samples each of dimension D

        Args:
            X (ndarray of shape (N, D)): training data
            y (ndarray of shape (N,)): training labels; y[i] = c
          means that X[i] has label 0 <= c < C for C classes
            learning_rate (float): learning rate for optimization
            reg (float): regularization strength
            num_iters (integer): number of steps to take when optimizing
            batch_size (integer): number of training examples to use
                at each step
            verbose (boolean): If true, print progress during optimization
            is_apply_augmentation (boolean): If true, applies augmentation for
                each batch of X

        Returns:
            tuple:
            * loss history (list of floats): value of the loss function at
                each training iteration
            * weights history (list of ndarrays): values of weights at each
                training iteration
        """
        num_train, dim = X.shape
        num_classes = (
            np.max(y) + 1
        )  # assume y takes values 0...K-1 where K is number of classes
        if self.W is None:
            # lazily initialize W
            self.W = 0.001 * np.random.randn(dim, num_classes)

        # Run stochastic gradient descent to optimize W
        loss_history = []
        weights_history = []
        for it in range(num_iters):
            X_batch = None
            y_batch = None

            indices = np.random.choice(num_train, batch_size, replace=True)
            X_batch = X[indices]
            y_batch = y[indices]

            if is_apply_augmentation:
                X_batch = augmentation.flip_horizontal(X_batch)

            # evaluate loss and gradient
            loss, grad = self.loss(X_batch, y_batch, reg)
            loss_history.append(loss)

            # perform parameter update
            self.W -= learning_rate * grad
            weights_history.append(deepcopy(self.W))

            if verbose and it % 100 == 0:
                print("iteration %d / %d: loss %f" % (it, num_iters, loss))

        return loss_history, weights_history

    def predict(self, X):
        """
        Use the trained weights of this linear classifier to predict labels for
        data points.

        Inputs:
        - X: A numpy array of shape (N, D) containing training data; there are N
          training samples each of dimension D.

        Returns:
        - y_pred: Predicted labels for the data in X. y_pred is a 1-dimensional
          array of length N, and each element is an integer giving the predicted
          class.
        """
        y_pred = np.zeros(X.shape[0])

        scores = X.dot(self.W)
        y_pred = np.argmax(scores, axis=1)

        return y_pred
