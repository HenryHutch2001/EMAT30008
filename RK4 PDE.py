# %%
from scipy.integrate import solve_ivp
import math
from math import ceil
import numpy as np 
import matplotlib.pyplot as plt
import matplotlib.animation as animation 
from My_Functions import solve_toRK
# %% Discretising Space
D = 0.5 # Defining the diffusion coefficient
a = 0 # Defining the start of the domain 
b = 1 # Defining the end of the domain
alpha = 0 # Defining the boundary condition values for u(a,t) = alpha
beta = 0 # Defining the boundary condition values for u(b,t) = beta
N = 20 # Defining the number of gridpoints
GridSpace = np.linspace(a,b,N+1) #Defining the gridspace
x_int = GridSpace[1:-1] # Extracting the interior gridpoints from 'GridSpace'
dx = (b-a)/N # Defining 'Delta x'
C = 0.49 # C = dt*D/(dx)^2 must be =< 1/2
dt = (C*dx**2)/D # Rearranging the equation for C gives the equation for the time step
t_final = 1 # Final time in the simulation 
N_t = ceil(t_final/dt) # Determine the number of timesteps 
u = np.zeros((N_t+1,N-1))





