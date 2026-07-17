import numpy as np
import unittest
import HigherOrderRegression


class TestHigherOrderRegression(unittest.TestCase):
    def setUp(self):
        self.x = np.array([-3.23, 0., 10.9])
        self.y = np.array([1.0, 2.1, 5.02])
        self.testModel = HigherOrderRegression.HigherOrderRegression(
            self.x,
            self.y,
            self.x,
            self.y,
            3,
            False
        )

    def testPredict(self):
        self.testModel.weight = np.array([2., 3., -1.])
        self.testModel.bias = 2.
        np.testing.assert_almost_equal(
            self.testModel.predict(self.x),
            np.array([60.536967, 2., -914.799])
        )

        self.testModel.weight = np.array([-2.3, 4.1, -20.])
        self.testModel.bias = 23.01
        np.testing.assert_almost_equal(
            self.testModel.predict(self.y),
            [4.81, -148.959, -2415.33452]
        )

    def testCostFunction(self):
        self.testModel.weight = np.array([-1.2, 4, 0, ])
        self.testModel.bias = 0
        self.assertAlmostEqual(
            self.testModel.costFunction(self.x, self.y),
            70323.74252592002
        )

    def testGradient(self):
        self.testModel.weight = np.array([0.3, 0.1, -0.5])
        self.testModel.bias = -13.01
        np.testing.assert_almost_equal(
            self.testModel.gradient(self.x, self.y),
            [-4.41726718e+02, -4.73246634e+03, -5.14952375e+04, -5.61584414e+05],
            decimal=3
        )

    def testTrain(self):
        self.testModel.weight = np.array([0., 0., 0.])
        self.testModel.bias = 0.
        self.testModel.train(1000, 1e-6)
        np.testing.assert_almost_equal(
            self.testModel.weight,
            np.array([-0.00204883, 0.00979237, 0.00297441])
        )
        self.assertAlmostEqual(
            self.testModel.bias,
            0.0021237698058755006
        )

    def testRegularization(self):
        self.testModel.regularization = True
        self.testModel.regularizationParameter = 2.
        self.testModel.weight = np.array([2., 3., 1.])
        self.testModel.bias = -2.
        self.assertAlmostEqual(
            self.testModel.costFunction(self.x, self.y),
            925531.2881501429
        )
        np.testing.assert_almost_equal(
            self.testModel.gradient(self.x, self.y),
            np.array([1.10018629e+03, 1.21415410e+04, 1.31906751e+05, 1.43882232e+06]),
            decimal=2
        )
        self.testModel.train(10000, 1e-6)
        self.assertAlmostEqual(
            self.testModel.costFunction(self.x, self.y),
            40.603441550688316
        )


unittest.main()    # Run the unittest.   