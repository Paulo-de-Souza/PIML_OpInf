import torch
from torch.autograd import grad
import matplotlib.pyplot as plt

dtype = torch.float
device = torch.device("cpu")


def generateGrid1d(length, samples=20, initial_coordinate=0.0, midpoints=False):
    """Generate an evenly space grid of a given length and a given number of samples."""

    # Generate the grid
    if midpoints is False:
        x = torch.linspace(initial_coordinate, initial_coordinate + length, samples, requires_grad=True)
    else:
        dx = length / (samples - 1)
        start = initial_coordinate + dx / 2
        end = initial_coordinate + length - dx / 2

        x = torch.linspace(start, end, samples - 1, requires_grad=True)

    # Reshape on a column tensor and return
    return x.view(-1, 1)


# def trapezoidalIntegration1d(y, x):
#    """Compute the integral of y = f(x) over the range of x using the trapezoidal rule."""
        # Your code goes here.


# def midpointIntegration1d(y, x):
#     """Compute the integral of y = f(x) over the range of x using the midpoint rule."""
        # Your code goes here.


def getDerivative(y, x, n):
    """Compute the nth order derivative of y = f(x) with respect to x."""

    if n == 0:
        return y
    else:
        dy_dx = grad(y, x, torch.ones(x.size()[0], 1, device=device), create_graph=True, retain_graph=True)[0]
        return getDerivative(dy_dx, x, n - 1)
    

def plotDisplacementsBar(x, u, u_analytic=None):
    """Plot displacements."""

    # Set up plot
    fig, ax = plt.subplots()
    ax.set_title("Displacements")
    ax.set_xlabel("$x$")
    ax.set_ylabel("$u(x)$")

    # Plot data
    if u_analytic != None:
        ax.plot(x.detach().numpy(), u_analytic(x.detach().numpy()), linewidth=2, label="$u_{\mathrm{analytic}}$")
    ax.plot(x.detach().numpy(), u.detach().numpy(), color='k', linestyle=':', linewidth=5, label="$u_{\mathrm{pred}}$")
    ax.legend()
    plt.show()
    fig.tight_layout()


def plotStiffnessBar(x, EA, EA_analytic=None):
    """Plot stiffness."""

    # Set up plot
    fig, ax = plt.subplots()
    ax.set_title("Stiffness")
    ax.set_xlabel("$x$")
    ax.set_ylabel("$EA$")

    # Plot data
    ax.plot(x.detach().numpy(), EA.detach().numpy(), label="$EA_{pred}$")
    if EA_analytic != None:
        ax.plot(x.detach().numpy(), EA_analytic(x.detach().numpy()), label="$EA_{analytic}$")

    ax.legend()
    plt.show()
    fig.tight_layout()

