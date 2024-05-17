import math
import random

from back.sigmoid import Sigmoid


class NeuralNetwork:
    def __init__(self, input_size, hidden_layer_sizes, output_size):
        self.input_size = input_size
        self.hidden_layer_sizes = hidden_layer_sizes
        self.output_size = output_size
        self.sigmoid = Sigmoid()
        #pesos,  y sesgos aleatorios
        self.weights = []
        self.biases = []
        self.threshold = []

        layer_sizes = [input_size] + hidden_layer_sizes + [output_size]

        # Capa de entrada a la primera capa oculta
        self.weights.append(
            [[random.uniform(0, 1) for _ in range(self.input_size)] for _ in range(self.hidden_layer_sizes[0])])
        self.threshold.append([random.uniform(0, 1) for _ in range(self.hidden_layer_sizes[0])])

        for valor in self.threshold:
            print(valor)
            #self.biases.append(valor * -1)

        # Capas ocultas
        for i in range(1, len(self.hidden_layer_sizes)):
            self.weights.append(
                [[random.uniform(-1, 1) for _ in range(self.hidden_layer_sizes[i - 1])] for _ in
                 range(self.hidden_layer_sizes[i])])
            self.biases.append([random.uniform(-1, 1) for _ in range(self.hidden_layer_sizes[i])])

        # Capa oculta a la capa de salida
        self.weights.append(
            [[random.uniform(-1, 1) for _ in range(self.hidden_layer_sizes[-1])] for _ in range(self.output_size)])
        self.biases.append([random.uniform(-1, 1) for _ in range(self.output_size)])

    def dot_product(self, a, b):
        total = 0
        for x, y in zip(a, b):
            if isinstance(x, (int, float)) and isinstance(y, (int, float)):
                total += x * y
        return total

    def forward(self, x):
        #propagación hacia adelante
        activations = [x]  #lista de activación con el vector de entrada
        zs = []

        for i in range(len(self.weights)):
            #z = x1*w1 + x2*w2 + bia (sesgo)
            z = [self.dot_product(row, activations[-1]) + bias for row, bias in zip(self.weights[i], self.biases[i])]
            a = [self.sigmoid.sigmoid(x) for x in z]
            activations.append(a)
            zs.append(z)

        return activations[-1], zs

    def backprop(self, x, y, learning_rate):
        m = len(x)
        activations, zs = self.forward(x)

        # Calcular el error en la capa de salida
        errors = [y[i] - activations[-1][i] for i in range(len(y))]
        #errors = [(activations[-1][i] - y[i]) * self.sigmoid.sigmoid(zs[-1][i]) * self.sigmoid.sigmoid_prime(zs[-1][i])
         #         for i in
          #        range(self.output_size)]

        # Actualizar los pesos y sesgos de la última capa
        self.weights[-1] = [[self.weights[-1][j][k] - learning_rate * errors[j] * activations[-2][k] / m for k in
                             range(self.hidden_layer_sizes[-1])] for j in range(self.output_size)]
        self.biases[-1] = [self.biases[-1][j] - learning_rate * errors[j] / m for j in range(self.output_size)]

        # Propagar hacia atrás el error y actualizar los pesos y sesgos de las capas anteriores
        for i in range(len(self.weights) - 2, -1, -1):
            delta = [sum(self.weights[i + 1][k][j] * errors[k] for k in range(
                self.output_size if i == len(self.weights) - 2 else self.hidden_layer_sizes[i + 1])) * self.sigmoid.sigmoid(
                zs[i][j]) * (1 - self.sigmoid.sigmoid(zs[i][j])) for j in range(self.hidden_layer_sizes[i])]
            self.weights[i] = [[self.weights[i][j][k] - learning_rate * delta[j] * activations[i][k] / m for k in
                                range(self.hidden_layer_sizes[i - 1] if i > 0 else self.input_size)] for j in
                               range(self.hidden_layer_sizes[i])]
            self.biases[i] = [self.biases[i][j] - learning_rate * delta[j] / m for j in range(self.hidden_layer_sizes[i])]
            errors = delta

    def train(self, x, y, epochs, learning_rate):
        for epoch in range(epochs):
            self.backprop(x, y, learning_rate)

    def predict(self, x):
        activations, _ = self.forward(x)
        return [round(x) for x in activations]
