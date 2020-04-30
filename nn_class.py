import sys
import numpy as np

class NeuralNet():
    def __init__(self, input_size, output_size, hidden_layer_sizes=[], rate=0.5):
        self.input_size = input_size
        self.output_size = output_size
        self.num_hidden_layers = len(hidden_layer_sizes)
        self.hidden_layer_sizes = hidden_layer_sizes
        sizes = [input_size] + hidden_layer_sizes + [output_size]
        self.weights = []
        self.biases = []
        for i in range(1, len(sizes)):
            self.weights.append(np.random.random((sizes[i-1], sizes[i])))
            self.biases.append(np.random.random(sizes[i]))

        self.rate = rate

    def sigmoid(self, x):
        return 1/(1 + np.exp(np.negative(x)))

    def sigmoid_prime(self, x):
        return self.sigmoid(x) * (1 - self.sigmoid(x))

    def loss(self, gold, prediction):
        return np.sum(- gold*np.log(prediction) - (np.negative(gold) + 1)*np.log(np.negative(prediction) + 1))

    def loss_prime(self, gold, prediction):
        return - gold/prediction + (np.negative(gold) + 1)/(np.negative(prediction) + 1)

    def forward(self, inputs, trace=False):
        if trace:
            products = []
            activations = [inputs]
        output = inputs
        for i in range(len(self.weights)):
            product = np.dot(output, self.weights[i]) + self.biases[i]
            output = self.sigmoid(product)
            if trace:
                products.append(product)
                activations.append(output)
        if trace:
            return (output, products, activations)
        return output

    def train(self, data, epochs=1000, cooling=0.99):
        self.rate = self.rate*cooling
        for itr in range(epochs):
            l = 0
            for inputs, labels in data:
                bias_error = [np.zeros(bias.shape) for bias in self.biases]
                weight_error = [np.zeros(weight.shape) for weight in self.weights]
                output, products, activations = self.forward(inputs, trace=True)
                l += self.loss(labels, output)
                l_p = self.loss_prime(labels, output)
                delta = l_p * self.sigmoid_prime(products[-1])
                bias_error[-1] = delta
                weight_error[-1] = np.dot(delta.transpose(), activations[-2]).transpose()
                for i in reversed(range(self.num_hidden_layers)):
                    delta = np.dot(self.weights[i+1], delta.transpose()).transpose() * self.sigmoid_prime(products[i])
                    bias_error[i] = delta
                    weight_error[i] = np.dot(delta.transpose(), activations[i]).transpose()

                for i in range(self.num_hidden_layers + 1):
                    self.weights[i] = self.weights[i] - self.rate*weight_error[i]
                    self.biases[i] = self.biases[i] - self.rate*bias_error[i]

            #print("loss: {0}".format(l))
