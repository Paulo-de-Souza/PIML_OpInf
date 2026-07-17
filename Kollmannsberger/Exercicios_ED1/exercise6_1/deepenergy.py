import torch
import numpy as np
import matplotlib.pyplot as plt
import utilities


class DeepEnergyBarModel:
    """A class used for the definition of Deep Energy Models for one dimensional bars."""

    def __init__(self, E, A, L, dirichlet_bc, neumann_bc=None, dist_load=None,
                 integrator=utilities.trapezoidalIntegration1d):
        """Construct a DeepEnergyBarModel model"""

        self.E = E
        self.A = A
        self.L = L
        self.dirichlet_bc = dirichlet_bc
        self.dist_load = dist_load
        self.neumann_bc = neumann_bc
        self.model = self.buildModel(1, [20], 1)
        self.internal_energy_history = None
        self.external_energy_history = None
        self.potential_energy_history = None
        self.optimizer = None
        self.integrator = integrator
        self.loss = None
    
    def buildModel(self, input_dimension, hidden_dimension, output_dimension):
        """Build a neural network of given dimensions."""
    
        modules = []
        modules.append(torch.nn.Linear(input_dimension, hidden_dimension[0]))
        modules.append(torch.nn.Tanh())
        for i in range(len(hidden_dimension) - 1):
            modules.append(torch.nn.Linear(hidden_dimension[i], hidden_dimension[i + 1]))
            modules.append(torch.nn.Tanh())
    
        modules.append(torch.nn.Linear(hidden_dimension[-1], output_dimension))
    
        model = torch.nn.Sequential(*modules)
    
        return model

#     def getDisplacements(self, x):
#         """Get displacements with enforced boundary conditions."""
        # Your code goes here.

#     def getStrains(self, u, x):
#         """Compute strains for given displacements u."""
        # Your code goes here.

#     def getInternalEnergy(self, u, x):
#         """Compute the internal energy for given displacements."""
        # Your code goes here.

#     def getExternalEnergy(self, u, x):
#         """Compute the external energy of the acting loads for given displacements."""
        # Your code goes here.

#     def getEnergyValues(self, u, x):
#         """Compute the internal, external and potential energy for given displacements."""
        # Your code goes here.
#         return internal_energy, external_energy, potential_energy

    def closure(self):
        """Closure function for the optimizer."""
        self.optimizer.zero_grad()
        u_pred = self.getDisplacements(self.x)
        energy = self.getEnergyValues(u_pred, self.x)
        loss = energy[2]
        loss.backward(retain_graph=True)
        return loss   

    def train(self, epochs, optimizer, samples=50, **kwargs):
        """Train the model."""
        # Generate grid
        if self.integrator is utilities.midpointIntegration1d:
            x = utilities.generateGrid1d(self.L, samples, midpoints=True)
        else:
            x = utilities.generateGrid1d(self.L, samples)
        self.x = x

        # Set optimizer
        if optimizer=='Adam':
            self.optimizer = torch.optim.Adam(self.model.parameters(), **kwargs)
        elif optimizer=='LBFGS':
            self.optimizer = torch.optim.LBFGS(self.model.parameters(), **kwargs)

        # Initialize history arrays
        self.internal_energy_history = np.zeros(epochs)
        self.external_energy_history = np.zeros(epochs)
        self.potential_energy_history = np.zeros(epochs)

        # Training loop
        for i in range(epochs):
            # Predict displacements
            u_pred = self.getDisplacements(x)

            # Get energy values, where the potential energy is the cost function
            internal_energy, external_energy, potential_energy = self.getEnergyValues(u_pred, x)

            # Add energy values to history
            self.internal_energy_history[i] += internal_energy.item()
            self.external_energy_history[i] += external_energy.item()
            self.potential_energy_history[i] += potential_energy.item()

            # Print training state
            self.printTrainingState(i, epochs)

            # Set gradients to zero
            self.optimizer.zero_grad()

            # Compute gradient (backwardpropagation)
            potential_energy.backward(retain_graph=True)

            # Update parameters
            self.optimizer.step(self.closure)
            
        self.x = None
    def printTrainingState(self, epoch, epochs, print_every=100):
        """Print the energy values of the current epoch in a training loop."""

        # Check whether there is data available
        if self.internal_energy_history is None or \
                self.external_energy_history is None or \
                self.potential_energy_history is None:
            print("There is no data to print.")

        elif epoch == 0 or epoch == (epochs - 1) or epoch % print_every == 0 or print_every == 'all':
            # Prepare string
            string = "Epoch: {}/{}\t\tInternal energy = {:2f}\t\tExternal energy = {:2f}\t\tPotential energy = {:2f}"

            # Format string and print
            print(string.format(epoch, epochs - 1, self.internal_energy_history[epoch],
                                self.external_energy_history[epoch], self.potential_energy_history[epoch]))

    def plotTrainingHistory(self, yscale='linear'):
        """Plot the training history."""

        # Set up plot
        fig, ax = plt.subplots()
        ax.set_title("Energy history")
        ax.set_xlabel("Epochs")
        ax.set_ylabel("Energy $\Pi_{i}$, $\Pi_{e}$, $\Pi_{\mathrm{tot}}$")
        plt.yscale(yscale)

        # Plot data
        ax.plot(self.internal_energy_history, color='b', linestyle='--', linewidth=2, label="Internal energy")
        ax.plot(self.external_energy_history, color='r', linestyle='-.', linewidth=2, label="External energy")
        ax.plot(self.potential_energy_history, 'k', linewidth=2, label="Potential energy")

        ax.legend()
        plt.show()

        fig.tight_layout()
