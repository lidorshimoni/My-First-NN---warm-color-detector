import numpy as np


class NeuralNetwork():

    def __init__(self):
        np.random.seed(1)

        # Set synaptic weights to a 3x1 matrix,
        # with values from -1 to 1 and mean 0
        self.synaptic_weights = np.matrix([[2.0], [0.0], [-2.0]])

    def sigmoid(self, x):
        # normalaize func
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        # for weights adjustment
        return x * (1 - x)

    def super_train(self, super_training_inputs, super_training_outputs, super_training_iterations):
        # super_train for the given numbers
        for iteration in range(super_training_iterations):
            # Pass super_training set through the neural network
            output = self.think(super_training_inputs)
            # Calculate the error
            error = super_training_outputs - output
            # Less confident weights are adjusted more? idk
            adjustments = np.dot(super_training_inputs.T, error * self.sigmoid_derivative(output))
            # Adjust synaptic weights
            self.synaptic_weights += adjustments

    def think(self, inputs):
        # calculate the outputs
        inputs = inputs.astype(float)
        output = self.sigmoid(np.dot(inputs, self.synaptic_weights))
        return output
