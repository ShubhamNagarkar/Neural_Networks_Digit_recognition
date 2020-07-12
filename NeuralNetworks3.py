import random
import numpy as np
import csv
import sys


class NeuralNetwork:
    def __init__(self, size):
        self.activation = []
        self.z = []
        self.input = []
        self.output = [[0], [0], [0], [0], [0], [0], [0], [0], [0], [0]]
        self.layers = len(size)
        self.weights = [np.random.randn(y, x) * np.sqrt(2 / x) for x, y in zip(size[:-1], size[1:])]
        self.bias = [np.random.randn(y, 1) for y in size[1:]]

    def ini_output(self):
        self.output = [[0], [0], [0], [0], [0], [0], [0], [0], [0], [0]]

    def set_output(self, output_layer):
        self.output[output_layer] = [1]

    def Sigmoid(self, z):
        sig_val = (1.0 / (1.0 + np.exp(-z)))
        return sig_val

    def D_SoftMax(self, z):
        np.diag(z)
        z_vector = z.reshape(z.shape[0], 1)
        z_matrix = np.tile(z_vector, z.shape[0])
        derivative = np.diag(z) - (z_matrix * np.transpose(z_matrix))
        return derivative

    def Relu(self, z):
        return np.maximum(0, z)

    def D_Relu(self, z):
        z_prime = []
        for x in z:
            if x[0] <= 0:
                z_prime.append([0])
            else:
                z_prime.append([1])
        return z_prime

    def D_Sigmoid(self, z):
        return self.Sigmoid(z) * (1 - self.Sigmoid(z))

    def SoftMax(self, z):
        e_x = np.exp(z - np.max(z))
        a = e_x / e_x.sum(axis=0)
        return a

    def FeedForward(self, a):
        self.activation = [a]
        cnt = 0
        for bias, weight in zip(self.bias, self.weights):
            z = np.dot(weight, a) + bias
            self.z.append(z)
            if cnt < self.layers - 2:
                a = self.Relu(z)
            else:
                a = self.SoftMax(z)
            self.activation.append(a)
            cnt += 1
        return a

    def D_CrossEntropy(self, a, y):
        return a - y

    def convert_to_batches(self, x_train, y_train, epoch, batch_size, alpha):
        data = [(x, y) for x, y in zip(x_train, y_train)]
        for epochs in range(epoch):
            random.shuffle(data)
            batches = [data[indices: indices + batch_size] for indices in range(0, len(data), batch_size)]
            for batch in batches:
                self.update_w_b(batch, alpha)

    def update_w_b(self, batch, alpha):
        batch_w = [np.zeros(w.shape) for w in self.weights]
        batch_b = [np.zeros(b.shape) for b in self.bias]
        for x, y in batch:
            self.ini_output()
            self.set_output(y)
            self.activation = []
            self.z = []
            self.FeedForward(x)
            Dc_b, Dc_w = self.BackProp(self.output)
            batch_w = [bw + d_w for bw, d_w in zip(batch_w, Dc_w)]
            batch_b = [bb + d_b for bb, d_b in zip(batch_b, Dc_b)]
        self.weights = [w - (alpha / len(batch)) * b_w for w, b_w in zip(self.weights, batch_w)]
        self.bias = [b - (alpha / len(batch)) * b_b for b, b_b in zip(self.bias, batch_b)]

    def BackProp(self, y):
        Dc_b = [np.zeros(b.shape) for b in self.bias]
        Dc_w = [np.zeros(w.shape) for w in self.weights]

        Dc_a = self.D_CrossEntropy(self.activation[-1], y)
        delta = Dc_a
        Dc_b[-1] = delta
        Dc_w[-1] = np.dot(delta, self.activation[-2].transpose())
        for i in range(2, self.layers):
            z = self.z[-i]
            d_sigmoid = self.D_Relu(z)
            delta = np.dot(self.weights[-i + 1].transpose(), delta) * d_sigmoid
            Dc_b[-i] = delta
            Dc_w[-i] = np.dot(delta, np.transpose(self.activation[-i - 1]))
        return Dc_b, Dc_w

    def CrossEntropy(self, p_output):
        epsilon = 1e-5
        return -sum([self.output[i][0] * (np.log2(p_output[i][0] + epsilon)) for i in range(len(self.output))])

    def load_Data(self):
        if len(sys.argv) > 1:
            f1 = open(sys.argv[1])
            f2 = open(sys.argv[2])
            f3 = open(sys.argv[3])
        else:
            f1 = open("train_image.csv")
            f2 = open("train_label.csv")
            f3 = open("test_image.csv")

        csv_f1 = csv.reader(f1)
        csv_f2 = csv.reader(f2)
        csv_f3 = csv.reader(f3)

        x_train = []
        y_train = []
        x_test = []

        for img, op in zip(csv_f1, csv_f2):
            x = np.asarray(img, dtype=np.float32).reshape(784, 1)
            x = np.divide(x, 256 * 1.0)
            y_train.append(int(op[0]))
            x_train.append(x)

        for img in csv_f3:
            x = np.asarray(img, dtype=np.float32).reshape(784, 1)
            x = np.divide(x, 256 * 1.0)
            x_test.append(x)

        return x_train, y_train, x_test

    def test_predictions(self, data):
        results = []
        for x_t, y_t in data:
            results.append((np.argmax(self.FeedForward(x_t)), y_t))
        return sum(int(x == y) for (x, y) in results)

    def create_predictions(self, x_test):
        f = open("test_predictions.csv", 'w')
        for img in x_test:
            f.write(str(np.argmax(self.FeedForward(img))) + "\n")
        f.close()


nn = NeuralNetwork([784, 64, 64, 64, 10])
x_train, y_train, x_test = nn.load_Data()
nn.convert_to_batches(x_train, y_train, 50, 8, 0.08)
nn.create_predictions(x_test)
