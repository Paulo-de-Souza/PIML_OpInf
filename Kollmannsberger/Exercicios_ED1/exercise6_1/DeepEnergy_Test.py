from deepenergy import DeepEnergyBarModel
import numpy as np
import torch
import unittest
import utilities

class TestDeepEnergy(unittest.TestCase):

    def setUp(self):
        self.E = lambda x: 2.
        self.A = lambda x: x + 1.
        self.L = 3.
        self.dirichlet_bc = 'fixed_right'
        self.neumann_bc = [[0, 0]]
        self.dist_Load = lambda x: x ** 2
        self.testModel = DeepEnergyBarModel(self.E,
                                            self.A,
                                            self.L,
                                            dirichlet_bc=self.dirichlet_bc,
                                            neumann_bc=self.neumann_bc,
                                            dist_load=self.dist_Load)

    def testTrapezoidalIntegration1d(self):
        x = torch.linspace(0, 10, 100)
        y = x ** 2 + 3
        self.assertAlmostEqual(float(utilities.trapezoidalIntegration1d(y, x)),
                               363.3503,
                               places=4)

    def testmidpointIntegration1d(self):
        x = torch.linspace(-10, 10, 200)
        y = torch.sin(x) * x - 1.2
        self.assertAlmostEqual(float(utilities.trapezoidalIntegration1d(y, x)),
                               -8.321696281433105,
                               places=4)

    def testGetDisplacements(self):
        x = utilities.generateGrid1d(self.L)
        self.testModel.dirichlet_bc = 'fixed_left'
        u = self.testModel.getDisplacements(x)
        self.assertAlmostEqual(float(u[0]), 0, places=5)

        self.testModel.dirichlet_bc = 'fixed_right'
        u = self.testModel.getDisplacements(x)
        self.assertAlmostEqual(float(u[-1]), 0, places=5)

        self.testModel.dirichlet_bc = 'fixed_both'
        u = self.testModel.getDisplacements(x)
        self.assertAlmostEqual(float(u[0]), 0, places=5)
        self.assertAlmostEqual(float(u[-1]), 0, places=5)

        self.testModel.model[0].weight.data = torch.ones(20, 1) * 2
        self.testModel.model[2].weight.data = torch.ones(1, 20) * 0.2
        self.testModel.model[0].bias.data = torch.ones(20) * 3
        self.testModel.model[2].bias.data = torch.ones(1) * 5

        u = torch.tensor([[0.0000],
                          [4.0341],
                          [7.6241],
                          [10.7665],
                          [13.4602],
                          [15.7049],
                          [17.5005],
                          [18.8471],
                          [19.7449],
                          [20.1938],
                          [20.1938],
                          [19.7451],
                          [18.8476],
                          [17.5014],
                          [15.7064],
                          [13.4626],
                          [10.7701],
                          [7.6288],
                          [4.0388],
                          [0.0000]], )
        np.testing.assert_almost_equal(u.tolist(),
                                       self.testModel.getDisplacements(x).detach().tolist(),
                                       decimal=4)

    def testGetStrains(self):
        x = utilities.generateGrid1d(200)
        u = torch.cos(x) - x ** 3 + 32 * x
        e = -torch.sin(x) - 3 * x ** 2 + 32
        np.testing.assert_almost_equal(e.tolist(),
                                       self.testModel.getStrains(u, x).detach().tolist(),
                                       decimal=4)

    def testGetInternalEnergy(self):
        x = utilities.generateGrid1d(40)
        u = 0.002 * x ** 5 - 40 * x ** 2 - 0.1 * x
        internal_energy = 75322802176.0
        self.assertAlmostEqual(internal_energy,
                               self.testModel.getInternalEnergy(u, x),
                               places=4)

    def testGetExternalEnergy(self):
        self.testModel.neumann_bc = []
        x = utilities.generateGrid1d(60)
        u = 1 - torch.sin(x)
        external_energy = -72234.9375
        self.assertAlmostEqual(external_energy,
                               self.testModel.getExternalEnergy(u, x).detach().numpy()[0],
                               places=0)

        dist_load = lambda x: 0
        self.testModel.dist_load = dist_load
        self.testModel.dirichlet_bc = 'fixed_left'
        self.testModel.neumann_bc = [[10, -1]]
        u = x ** 2
        external_energy = -36000.00390625
        self.assertAlmostEqual(external_energy,
                               self.testModel.getExternalEnergy(u, x).detach().numpy()[0],
                               places=0)

    def testGetEnergyValues(self):
        dist_load = lambda x: x + 1
        self.testModel.dist_load = dist_load
        self.testModel.neumann_bc = [[5, -1]]
        self.testModel.dirichlet_bc = 'fixed_left'
        x = utilities.generateGrid1d(25)
        u = 0.001 * x ** 3 + x ** 2 - x + torch.sin(x)
        internal_energy = 415396.8438
        external_energy = -102730.5547
        potential_energy = 312666.2812
        energy_values = self.testModel.getEnergyValues(u, x)
        self.assertAlmostEqual(internal_energy,
                               float(energy_values[0]),
                               places=1)
        self.assertAlmostEqual(external_energy,
                               float(energy_values[1]),
                               places=1)
        self.assertAlmostEqual(potential_energy,
                               float(energy_values[2]),
                               places=1)


unittest.main()