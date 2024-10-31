from abc import ABCMeta, abstractmethod


class IClassifier(ABCMeta):

    @abstractmethod
    def train(self, X, y, hyperparams: dict):
        pass

    @abstractmethod
    def predict(self, X):
        pass

    @abstractmethod
    def __evaluate_loss(self, X, y, hyperparams: dict):
        pass
