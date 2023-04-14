# %%
import numpy as np
import matplotlib.pyplot as plt
from math import ceil
from My_Functions import CreateAandb
import math
# Define the parameters
a = 0
b = 1     # Length of the domain
D = 1
C = 0.5
t_final = 1.0     # Time to solve for
N = 100   # Number of grid points in the domain
GridSpace = np.linspace(a,b,N+1)
dx = (b-a)/N
dt = (C*dx**2)/D
N_t = ceil(t_final/dt) # Determine the number of timesteps 
bc_left = 0.0
bc_right = 0.0
# Calculate the grid spacing
# %%
def f(x,*args):
    return np.sin((math.pi*(x-a))/(b-a))

def source(x,t):
    return np.ones(np.size(x))
# %%
# Define the matrix for the implicit Euler method
A_dd,b_dd,x_int,dx = CreateAandb(N,a,b,'dirichlet',bc_left,bc_right)

u = np.zeros((N_t+1, N-1))
#u[0,:] = f(x_int,a,b)

M = np.identity(N-1)-C*A_dd
for n in range(0,N_t):
    b = u[n] + C*b_dd + dt*source(x_int, (n+1)*dt)
    u[n+1] = np.linalg.solve(M,b)

plt.plot(x_int,u[-1,:])
plt.show()




# %%
