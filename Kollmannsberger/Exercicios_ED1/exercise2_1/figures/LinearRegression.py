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

class LinearRegression:
    def __init__(self, x_train, y_train, x_test, y_test):
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test

        self.weight = 0.
        self.bias = 0.

    def predict(self, x):
        return self.weight * x + self.bias

    def costFunction(self, x, y):
        return np.average((self.predict(x) - y) ** 2)

    def gradient(self, x, y):    # Calculation of the gradients of the cost function.
        gradientBias = (2 * (self.predict(x) - y))    
        gradientWeight = (2 * (self.predict(x) - y)*x)
        return [np.average(gradientBias), np.average(gradientWeight)]

    def train(self, epochs, learningRate):    # Minimization of the cost function using gradient descent.
        for epoch in range(epochs):
            gradient = self.gradient(self.x_train, self.y_train)
            self.weight -= gradient[1] * learningRate
            self.bias -= gradient[0] * learningRate
            if (epoch % 10 == 0):    # Output of the cost function for every 10th epoch.
                print("\nEpoch ", epoch, ": ")
                print("MSE training data: ", self.costFunction(self.x_train, self.y_train))
                print("MSE testing data: ", self.costFunction(self.x_test, self.y_test))

    def plotFittedModel(self, filename, option=False):
        x_min = min(np.append(self.x_train, self.x_test))    # Set up grid for fitted model.
        x_max = max(np.append(self.x_train, self.x_test))
        x_fit = np.linspace(x_min, x_max, 10)

        fig, ax = plt.subplots(figsize=(4,3))    # Set up the plot.
        
        ax.set_xlabel("$x$")
        ax.set_ylabel("$y$")

        msize=10
        if option==True:
            msize=10
        ax.plot(self.x_train, self.y_train, "r", label="training data", marker="o", markersize=msize, linestyle="None")
        
        if option==True:
            ax.plot(self.x_test, self.y_test, "silver", label="test data", marker="x", markersize=msize, linestyle="None")
        
        ax.plot(x_fit, self.predict(x_fit), "k", label="model", linewidth=2)
        
        if option==True:
            ax.plot(-1.4,self.predict(-1.4), "k", label="prediction",marker="+",markersize=16,linestyle="None")
            ax.text(-1.4-0.4,self.predict(-1.4)+1.5,"($x_{\mathrm{new}}$, $\hat{y}_{\mathrm{new}}$)")
            ax.plot([-1.4,-1.4],[self.predict(-1.4)+0.5,self.predict(-1.4)+1.3],"k",linewidth=1)
        
        ax.legend(loc='lower right')
        
        plt.gca().axes.get_yaxis().set_visible(False)
        plt.gca().axes.get_xaxis().set_visible(False)
        
        fig.tight_layout()
        plt.show()     # Plot the data.
        plt.savefig(filename + ".eps")
        plt.savefig(filename + ".png")



