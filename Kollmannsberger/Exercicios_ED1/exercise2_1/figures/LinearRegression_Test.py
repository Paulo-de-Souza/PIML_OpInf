import numpy as np
import unittest
import LinearRegression


class TestLinearRegression(unittest.TestCase):
    def setUp(self):
        self.x = np.array([-3.23, 0., 10.9])
        self.y = np.array([1.0, 2.1, 5.02])
        self.testModel = LinearRegression.LinearRegression(self.x, self.y, self.x, self.y)

    def testPredict(self):
        self.testModel.weight = 1
        self.testModel.bias = 2
        np.testing.assert_almost_equal(
            self.testModel.predict(self.x).tolist(),
            [-1.23, 2., 12.9]
        )

        self.testModel.weight = -2.3
        self.testModel.bias = 23.01
        np.testing.assert_almost_equal(
            self.testModel.predict(self.y).tolist(),
            [20.71, 18.18, 11.464]
        )

    def testCostFunction(self):
        self.testModel.weight = -1.2
        self.testModel.bias = 0
        self.assertAlmostEqual(
            self.testModel.costFunction(self.x, self.y),
            113.4304586666667
        )

    def testGradient(self):
        self.testModel.weight = 0.3
        self.testModel.bias = -13.01
        np.testing.assert_almost_equal(
            self.testModel.gradient(self.x, self.y),
            [-29.89933333333333, -75.00122]
        )

    def testTrain(self):
        self.testModel.weight = 0.
        self.testModel.bias = 0.
        self.testModel.train(1000, 1e-2)
        self.assertAlmostEqual(
            self.testModel.weight,
            0.28027867922848254
        )
        self.assertAlmostEqual(
            self.testModel.bias,
            1.9900874446197405
        )


unittest.main()    # Run the unittest. 