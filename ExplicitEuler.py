# %%
from scipy.integrate import solve_ivp
import math
from math import ceil
import numpy as np 
import matplotlib.pyplot as plt
import matplotlib.animation as animation 
from My_Functions import solve_toRK
# %%
D = 0.5 # Defining the diffusion coefficient
a = 0 # Defining the start of the domain 
b = 1 # Defining the end of the domain
alpha = 0 # Defining the boundary condition values for u(a,t) = alpha
beta = 0 # Defining the boundary condition values for u(b,t) = beta
# %%
def f(x,*args):
    return np.sin((math.pi*(x-a))/(b-a)) # Defining the initial condition
# %%
N = 20 # Defining the number of gridpoints
GridSpace = np.linspace(a,b,N+1) #Defining the gridspace
x_int = GridSpace[1:-1] # Extracting the interior gridpoints from 'GridSpace'
dx = (b-a)/N # Defining 'Delta x'
C = 0.5 # C = dt*D/(dx)^2 must be =< 1/2
# %%
dt = (C*dx**2)/D # Rearranging the equation for C gives the equation for the time step
t_final = 1 # Final time in the simulation 
N_t = ceil(t_final/dt) # Determine the number of timesteps 
t = dt * np.arange(N_t) # Determine the time point
# %%
u = np.zeros((N_t+1,N-1))
u[0,:] = f(x_int,a,b)

for n in range(0,N_t):
    for i in range(0,N-1):
        if i == 0:
            u[n+1,0] = u[n,0] + C * (alpha-2*u[n,0]+u[n,1])
        if 0 < i and i < N-2:
            u[n+1,i] = u[n,i] + C * (u[n,i+1] - 2 * u[n,i] + u[n,i-1])
        else:
            u[n+1,N-2] = u[n, N-2] + C * (beta - 2 * u[n, N-2] + u[n, N-3])
# %%

x = np.linspace(a,b,1000)
t = 0

# Calculate the true solution at each (x,t) pair
def true_solution(x, t):
    return np.exp(-(D*math.pi**2*t)/(b-a)**2) * np.sin((math.pi*(x-a))/(b-a))

u_true = true_solution(x, t)
plt.plot(x, u_true)
plt.plot(x_int,u[1,:],'o')
plt.show()

# %%


# %%
