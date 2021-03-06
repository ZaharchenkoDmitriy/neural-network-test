import numpy as np

class Neuro(object):
    def __init__(self, learning_rate = 0.1):
        self.weights_0_1 = np.random.normal(0.0, 2 ** -0.5, (2, 3))
        self.weights_1_2 = np.random.normal(0.0, 1, (1, 2))
        self.sigmoid_mapper = np.vectorize(self.sigmoid)
        self.learning_rate = np.array(learning_rate)

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def layer(self, weights, input_data):
        inputs = np.dot(weights, input_data)
        return self.sigmoid_mapper(inputs)

    def predict(self, inputs):
        outputs_1 = self.layer(self.weights_0_1, inputs)
        outputs_2 = self.layer(self.weights_1_2, outputs_1)

        return outputs_2

    def gradient_layer(self, value):
        return value * (1 - value)

    def delta(self, error_layer, gradient_layer):
        return error_layer * gradient_layer

    def train(self, inputs, expected_value):
        outputs_1 = self.layer(self.weights_0_1, inputs)
        outputs_2 = self.layer(self.weights_1_2, outputs_1)

        actual_value = outputs_2[0]

        error_layer_2 = np.array([actual_value - expected_value])
        weights_delta_layer_2 = self.delta(error_layer_2, self.gradient_layer(actual_value))
        self.weights_1_2 -= (np.dot(weights_delta_layer_2, outputs_1.reshape(1, len(outputs_1)))) * self.learning_rate

        error_layer_1 = weights_delta_layer_2 * self.weights_1_2
        weights_delta_layer_1 = self.delta(error_layer_1, self.gradient_layer(outputs_1))
        self.weights_0_1 -= np.dot(inputs.reshape(len(inputs), 1), weights_delta_layer_1).T * self.learning_rate

        return actual_value

