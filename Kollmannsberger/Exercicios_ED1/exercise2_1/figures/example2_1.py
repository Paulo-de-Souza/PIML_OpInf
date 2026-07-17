import numpy as np
import LinearRegression


np.random.seed(100)    # Generate the data.
x_train = np.random.randn(20)
y_train = 2 * x_train + 3 + np.random.randn(20)*0.5
x_test = np.random.randn(20)
y_test = 2 * x_test + 3 + np.random.randn(20)*0.5


LinearModel = LinearRegression.LinearRegression(x_train, y_train, x_test, y_test)    # Create linear regression model.
LinearModel.weight=-0.5
LinearModel.bias=2.5

LinearModel.plotFittedModel("LinearRegressionCase0")
LinearModel.train(100, 1e-1)
LinearModel.plotFittedModel("LinearRegressionCase1")
LinearModel.plotFittedModel("LinearRegressionCase2",option=True)




