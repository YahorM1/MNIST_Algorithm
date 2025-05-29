import numpy as np

class NeuralNetwork2:

    def __init__(self, input_nodes, hidden_nodes, output_nodes, learning_rate):
        self.input_nodes = input_nodes
        self.hidden_nodes = hidden_nodes
        self.output_nodes = output_nodes

        self.W2 = np.random.randn(self.input_nodes, self.hidden_nodes) / np.sqrt(self.input_nodes / 2)
        self.b2 = np.random.rand(self.hidden_nodes)

        self.W3 = np.random.randn(self.hidden_nodes, self.output_nodes) / np.sqrt(self.hidden_nodes / 2)
        self.b3 = np.random.rand(self.output_nodes)

        self.Z3 = np.zeros([1, output_nodes])
        self.A3 = np.zeros([1, output_nodes])

        self.Z2 = np.zeros([1, hidden_nodes])
        self.A2 = np.zeros([1, hidden_nodes])

        self.Z1 = np.zeros([1, input_nodes])
        self.A1 = np.zeros([1, input_nodes])

        self.learning_rate = learning_rate

    def sigmoid(self, x):
        return 1. / (1 + np.exp(-x))

    def feed_forward(self):
        delta = 1e-7
        self.Z1 = self.input_data
        self.A1 = self.input_data

        self.Z2 = np.dot(self.A1, self.W2) + self.b2
        self.A2 = self.sigmoid(self.Z2)

        self.Z3 = np.dot(self.A2, self.W3) + self.b3
        self.A3 = self.sigmoid(self.Z3)

        return -np.sum(
            self.target_data * np.log(self.A3 + delta) + (1 - self.target_data) * np.log((1 - self.A3) + delta))

    def loss_val(self):
        delta = 1e-7

        self.Z1 = self.input_data
        self.A1 = self.input_data

        self.Z2 = np.dot(self.A1, self.W2) + self.b2
        self.A2 = self.sigmoid(self.Z2)

        self.Z3 = np.dot(self.A2, self.W3) + self.b3
        self.A3 = self.sigmoid(self.Z3)

        return -np.sum(
            self.target_data * np.log(self.A3 + delta) + (1 - self.target_data) * np.log((1 - self.A3) + delta))

    def train(self, input_data, target_data):

        self.target_data = target_data
        self.input_data = input_data

        loss_val = self.feed_forward()
        loss_3 = (self.A3 - self.target_data) * self.A3 * (1 - self.A3)
        self.W3 = self.W3 - self.learning_rate * np.dot(self.A2.T, loss_3)
        self.b3 = self.b3 - self.learning_rate * loss_3
        loss_2 = np.dot(loss_3, self.W3.T) * self.A2 * (1 - self.A2)
        self.W2 = self.W2 - self.learning_rate * np.dot(self.A1.T, loss_2)
        self.b2 = self.b2 - self.learning_rate * loss_2

    def predict(self, input_data):

        Z2 = np.dot(input_data, self.W2) + self.b2
        A2 = self.sigmoid(Z2)
        Z3 = np.dot(A2, self.W3) + self.b3
        A3 = self.sigmoid(Z3)
        predicted_num = np.argmax(A3)
        return predicted_num

    def accuracy(self, test_data):
        matched_list = []
        not_matched_list = []
        for index in range(len(test_data)):

            label = int(test_data[index, 0])
            data = (test_data[index, 1:] / 255.0 * 0.99) + 0.01
            predicted_num = self.predict(np.array(data, ndmin=2))
            if label == predicted_num:
                matched_list.append(index)
            else:
                not_matched_list.append(index)
        print("Current Accuracy = ", 100 * (len(matched_list) / (len(test_data))), " %")

        return matched_list, not_matched_list


