import numpy as np
import matplotlib.pyplot as plt

###########################
import matplotlib
import matplotlib.font_manager
matplotlib.rcParams["figure.dpi"] = 300
from matplotlib import rc
rc('font',**{'family':'serif','serif':['Computer Modern Roman'],
             'size' : 14})
rc('text', usetex=True)
############################

class HigherOrderRegression:
    def __init__(
            self,
            x_train,
            y_train,
            x_test,
            y_test,
            polynomialDegree,
            regularization='False',
            regularizationParameter=1
    ):

        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test
        self.polynomialDegree = polynomialDegree
        self.regularization = regularization
        self.regularizationParameter = regularizationParameter

        self.weight = np.zeros(polynomialDegree)    # Initialization of weights and bias
        self.bias = 0.

    def predict(self, x):
        result = 0
        for i in range(self.polynomialDegree):
            result += self.weight[i] * x ** (i + 1)
        return result + self.bias

    def costFunction(self, x, y):    # Calculation of the cost function
        if self.regularization == 'False':    # without regularization
            return np.average((self.predict(x) - y) ** 2)
        elif self.regularization == 'True':    # and with regularization.
            return np.average((self.predict(x) - y) ** 2) + self.regularizationParameter * np.sum(
                self.weight * self.weight)

    def gradient(self, x, y):    # Calculation of the gradient of the cost function.
        gradient = np.zeros([self.polynomialDegree + 1, len(x)])
        gradient[0] = (2 * (self.predict(x) - y))
        for i in range(1, self.polynomialDegree + 1):
            gradient[i] = gradient[i - 1] * x
        if self.regularization == 'True':
            for i in range(1, self.polynomialDegree + 1):
                gradient[i] += 2 * self.regularizationParameter * self.weight[i - 1]
        return np.average(gradient, axis=1)

    def train(self, epochs, learningRate):    # Minimization of the cost function using gradient descent.
        for epoch in range(epochs):
            gradient = self.gradient(self.x_train, self.y_train)
            self.bias -= gradient[0] * learningRate
            self.weight -= gradient[1::] * learningRate

            if (epoch % 10 == 0):
                print("\nEpoch ", epoch, ": ")
                print("Cost function training data: ", self.costFunction(self.x_train, self.y_train))
                print("Cost function testing data: ", self.costFunction(self.x_test, self.y_test))

    def plotFittedModel(self,filename):
        x_min = min(np.append(self.x_train, self.x_test))    # Set up grid for fitted model.
        x_max = max(np.append(self.x_train, self.x_test))
        x_fit = np.linspace(x_min, x_max, 100)

        fig, ax = plt.subplots(figsize=(4,3))    # Set up the plot.
        ax.set_xlabel("$x$")
        ax.set_ylabel("$y$")

        ax.plot(self.x_train, self.y_train, "r", label="training data", marker="o", markersize=10, linestyle="None")
        ax.plot(self.x_test, self.y_test, "silver", label="test data", marker="x", markersize=10, linestyle="None")
        ax.plot(x_fit, self.predict(x_fit), "k", label="model", linewidth=2)

        ax.legend(loc='lower right')
        
        plt.gca().axes.get_yaxis().set_visible(False)
        plt.gca().axes.get_xaxis().set_visible(False)
        plt.show()    # Plot the data.
        
        
        fig.tight_layout()
        plt.savefig(filename + ".eps")
        plt.savefig(filename + ".png")
        
    



