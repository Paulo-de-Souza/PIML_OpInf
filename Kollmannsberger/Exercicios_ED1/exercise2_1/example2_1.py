import numpy as np
import LinearRegression


np.random.seed(100)    # Generate the data.
x_train = np.random.randn(100)
y_train = 2 * x_train + 3 + np.random.randn(100)
x_test = np.random.randn(100)
y_test = 2 * x_test + 3 + np.random.randn(100)


LinearModel = LinearRegression.LinearRegression(x_train, y_train, x_test, y_test)    # Create linear regression model.

LinearModel.train(100, 1e-1)
LinearModel.plotFittedModel()
