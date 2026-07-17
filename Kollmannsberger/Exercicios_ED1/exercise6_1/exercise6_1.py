import utilities
from deepenergy import DeepEnergyBarModel
import numpy as np
import torch
import matplotlib.pyplot as plt

# Set a seed for the initilization of the weights and biases
torch.manual_seed(2)

# Manufactured solution
u_analytic = lambda x: -x**0.65 + 0.65 * x

# Problem data
E = lambda x: 1
A = lambda x: 1
L = 1
dirichlet_bc = 'fixed_left'
neumann_bc = [[0, -1]]
dist_load = lambda x: -0.2275 * x**-1.35

# Generate model
dem_model = DeepEnergyBarModel(E, A, L, dirichlet_bc, neumann_bc=neumann_bc, dist_load=dist_load,
                                integrator=utilities.midpointIntegration1d)

# Train model
#dem_model.train(epochs=4000, optimizer='Adam', lr=5e-3)
dem_model.train(epochs=100, optimizer='LBFGS', samples=200, lr=1e-3)

# Test model
samples = 100
x_test = utilities.generateGrid1d(L, samples)
u_test = dem_model.getDisplacements(x_test)

# Plot displacements
utilities.plotDisplacementsBar(x_test, u_test, u_analytic)

# Plot training history
dem_model.plotTrainingHistory()
