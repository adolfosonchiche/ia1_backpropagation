# This is a sample Python script.
import random

from back.neural_network import NeuralNetwork


# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.

def generar_sesgo_negativo():
  """Función para generar un sesgo aleatorio negativo entre 0 y -1."""
  return random.uniform(-1, 0)

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('PyCharm')
    sesgos = []
    sesgos.append([generar_sesgo_negativo() for _ in range(10)])
    print("\nSesgos:")
    print(sesgos)

    # Datos de entrenamiento
    X = [[0, 0], [0, 1], [1, 0], [1, 1]]
    y = [0, 1, 1, 0]

    # Crear la red neuronal
    nn = NeuralNetwork(input_size=2, hidden_layer_sizes=[4, 4], output_size=1)

    # Entrenar la red neuronal
    nn.train(X, y, epochs=10000, learning_rate=0.1)

    # Hacer predicciones
    print(nn.predict(X))


