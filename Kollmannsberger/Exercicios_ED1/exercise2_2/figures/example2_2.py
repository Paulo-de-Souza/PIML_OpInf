import numpy as np
import HigherOrderRegression


np.random.seed(10)    # Generate the data.
x_train = np.array([-1., -0.8, -0.2, 0.5, 0.7])
y_train = -x_train * x_train + 10.
y_train[3]-=0.2
x_test = np.array([-1.1, -0.6, 0.1, 0.3])
y_test = -x_test * x_test + 10.


polynomialDegreeList=[1,2,9]
regularizationList=['False','True']
regularizationParameterList=[1e0,1e-2,1e-8]
for j in range(2):
    if j==1:
        polynomialDegreeList=[9,9,9]
    for i in range(3):
        polynomialDegree = polynomialDegreeList[i]    # Create higher order regression model.
        HigherOrderModel = HigherOrderRegression.HigherOrderRegression(
            x_train,
            y_train,
            x_test,
            y_test,
            polynomialDegree,
            regularization = regularizationList[j],
            regularizationParameter = regularizationParameterList[i]
        )
        
        HigherOrderModel.train(10000, 1e-1)
        HigherOrderModel.plotFittedModel('HigherOrderRegressionCase'+str(i+j*3))
        
        






