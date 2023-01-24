import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse.linalg import eigsh
from scipy.sparse.linalg import eigs
from scipy import sparse

# Create grid of x and y coordinates ranging from 0 to 1
N = 150
X, Y = np.meshgrid(np.linspace(0,1,N, dtype=float), np.linspace(0,1,N, dtype=float))

def get_potential(x,y):
    return 0*x #-1/(x**2 + y**2 + 0.1**2)

V = get_potential(X, Y)

# Creating the Hamiltonian, H
diag = np.ones([N])
diags = np.array([diag, -2*diag, diag])
D = sparse.spdiags(diags, np.array([-1,0,1]), N, N)
T = -1/2 * sparse.kronsum(D, D)
U = sparse.diags(V.reshape(N**2), (0))
H = T + U

# Get eigenvectors and eigenvalues

eigenvalues, eigenvectors = eigsh(H, k=10, which='SM')

def get_e(n):
    return eigenvectors.T[n].reshape((N,N))

plt.figure(figsize=(9,9))
plt.contourf(X, Y, get_e(4)**2, 20)

plt.show()