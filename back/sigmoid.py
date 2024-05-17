import math


class Sigmoid:
    @staticmethod
    def sigmoid(x):
        return 1 / (1 + math.exp(-x))

    def sigmoid_prime(self, x):
        return self.sigmoid(x) * (1 - self.sigmoid(x))
