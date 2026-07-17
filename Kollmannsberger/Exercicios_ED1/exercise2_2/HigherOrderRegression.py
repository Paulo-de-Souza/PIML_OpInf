import numpy as np
import matplotlib.pyplot as plt


class HigherOrderRegression:
    def __init__(
            self,
            x_train,
            y_train,
            x_test,
            y_test,
            polynomialDegree,
            regularization=False,
            regularizationParameter=1
    ):

        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test
        self.polynomialDegree = polynomialDegree
        self.regularization = regularization
        self.regularizationParameter = regularizationParameter

        self.weight = np.zeros(polynomialDegree)
        self.bias = 0.

    # def predict(self, x):
        # Your code goes here.

    # def costFunction(self, x, y):
        # Your code goes here.
        
    # def gradient(self, x, y):
        # Your code goes here.

    # def train(self, epochs, learningRate):
        # Your code goes here.

    def plotFittedModel(self):
        x_min = min(np.append(self.x_train, self.x_test))    # Set up grid for fitted model.
        x_max = max(np.append(self.x_train, self.x_test))
        x_fit = np.linspace(x_min, x_max, 100)

        fig, ax = plt.subplots()    # Set up the plot.
        ax.set_xlabel("$x$")
        ax.set_ylabel("$y$")

        ax.plot(self.x_train, self.y_train, label="training data", marker="o", linestyle="None")
        ax.plot(self.x_test, self.y_test, label="testing data", marker="x", linestyle="None")
        ax.plot(x_fit, self.predict(x_fit), label="fitted model")

        ax.legend()
        plt.show()    # Plot the data.




