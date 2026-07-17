import numpy as np
import HigherOrderRegression


np.random.seed(10)    # Generate the data.
x_train = np.array([-1., -0.8, -0.2, 0.5, 0.7])
y_train = -x_train * x_train + 10.
y_train[3]-=0.2
x_test = np.array([-1.1, -0.6, 0.1, 0.3])
y_test = -x_test * x_test + 10.


polynomialDegree = 9    # Create higher order regression model.
HigherOrderModel = HigherOrderRegression.HigherOrderRegression(
    x_train,
    y_train,
    x_test,
    y_test,
    polynomialDegree,
    regularization = False,
    regularizationParameter = 1e-2
)

HigherOrderModel.train(10000, 1e-1)
HigherOrderModel.plotFittedModel()







