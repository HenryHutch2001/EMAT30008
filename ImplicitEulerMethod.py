# %%
import numpy as np
import matplotlib.pyplot as plt
from math import ceil

# Define the parameters
a = 0
b = 1     # Length of the domain
D = 1
C = 0.1
t_final = 1.0     # Time to solve for
N = 20   # Number of grid points in the domain
GridSpace = np.linspace(a,b,N+1)
x_internal = GridSpace[1:-1]
dx = (b-a)/N
dt = (C*dx**2)/D
N_t = ceil(t_final/dt) # Determine the number of timesteps 
bc_left = 1.0
bc_right = 0.0
# Calculate the grid spacing
# %%
def IC(x):
    return np.zeros(np.size(x))
# %%
# Define the matrix for the implicit Euler method
def CreateAandb(N,a,b,bc_left,bc_right):
    A = np.zeros((N-1,N-1))
    dx = (b-a)/N
    for i in range(0,N-1):
        A[i,i] = -2
    for i in range(0,N-2):
        A[i,i+1] = A[i,i-1] = 1
    B = np.zeros(N-1)
    B[0] = bc_left
    B[N-2] = bc_right
    return A,B,dx

A_dd,b_dd,dx = CreateAandb(N,a,b,bc_left,bc_right)
print(A_dd[1,0])

# %%
# Define the solution matrix and set the initial condition
u = np.zeros((N_t+1, N-1))
print(np.shape(u))
u0 = IC(x_internal)
u[0,:] = u0
print(len(u))
# %%
# Time stepping loop using the implicit Euler method
for n in range(1, N_t+1):
    u[:, n] = np.linalg.solve((np.identity(N_t+1)-C*A_dd)-C*b_dd, u[:, n-1])

# Plot the solution at the final time step
plt.plot(GridSpace, u[:, -1])
plt.xlabel('x')
plt.ylabel('u')
plt.title('Diffusion equation solution using the implicit Euler method')
plt.show()





# %%
