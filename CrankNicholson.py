# %%
# Obtained by taking an average of the implicit and explicit euler methods.
from My_Functions import CreateAandb
from math import ceil
import numpy as np
import matplotlib.pyplot as plt
import math

# %%
A_dd,b_dd,x_int,dx = CreateAandb(10,0,1,'robin',0.0,[1.0,0.0])

# %%


""" a = 0
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

A_dd,b_dd,x_int,dx = CreateAandb(N,a,b,'dirichlet',bc_left,bc_right)

u = np.zeros((N_t+1, N-1))

def f(x,*args):
    return np.sin((math.pi*(x-a))/(b-a))

def source(x,t):
    return np.ones(np.size(x))

LHS = ((np.identity(N-1))-((C/2)*A_dd))

for n in range(0,N_t):
    RHS = ((np.identity(N-1))+((C/2)*A_dd))@u[n] + C*b_dd + dt*source(x_int, (n+1)*dt)
    u[n+1] = np.linalg.solve(LHS,RHS)

plt.plot(x_int,u[-1,:])
plt.show() """

a = 0
b = 1

def f(x,*args):
    return np.sin((math.pi*(x-a))/(b-a))

def source(x,t):
    return np.ones(np.size(x))


def CrankNicholsonDirichlet(N,domain_start,domain_end,D,t_final,dt,bc_left,bc_right,initial_condition,source,*args):  
    a = domain_start
    b = domain_end
    A_dd,b_dd,x_int,dx = CreateAandb(N,a,b,'dirichlet',bc_left,bc_right)
    C = dt*D/(dx**2)
    N_t = ceil(t_final/dt) 
    u = np.zeros((N_t+1, N-1))
    u[0,:] = initial_condition(x_int,*args)
    LHS = ((np.identity(N-1))-((C/2)*A_dd))
    for n in range(0,N_t):
        RHS = ((np.identity(N-1))+((C/2)*A_dd))@u[n] + C*b_dd + dt*source(x_int, (n+1)*dt)
        u[n+1] = np.linalg.solve(LHS,RHS)
    return x_int,u[-1,:]

def CrankNicholsonNeumann(N,domain_start,domain_end,D,t_final,dt,bc_left,bc_right,initial_condition,source,*args):
    a = domain_start
    b = domain_end
    A_dd,b_dd,x_int,dx = CreateAandb(N,a,b,'neumann',bc_left,bc_right)
    C = dt*D/(dx**2)
    N_t = ceil(t_final/dt) 
    u = np.zeros((N_t+1, N))
    u[0,:] = initial_condition(x_int,*args)
    LHS = ((np.identity(N))-((C/2)*A_dd))
    for n in range(0,N_t):
        RHS = ((np.identity(N))+((C/2)*A_dd))@u[n] + C*b_dd + dt*source(x_int, (n+1)*dt)
        u[n+1] = np.linalg.solve(LHS,RHS)
    return x_int,u[-1,:]

def CrankNicholsonRobin(N,domain_start,domain_end,D,t_final,dt,bc_left,bc_right,initial_condition,source,*args):
    a = domain_start
    b = domain_end
    A_dd,b_dd,x_int,dx = CreateAandb(N,a,b,'robin',bc_left,bc_right)
    C = dt*D/(dx**2)
    N_t = ceil(t_final/dt) 
    u = np.zeros((N_t+1, N))
    u[0,:] = initial_condition(x_int,*args)
    LHS = ((np.identity(N))-((C/2)*A_dd))
    for n in range(0,N_t):
        RHS = ((np.identity(N))+((C/2)*A_dd))@u[n] + C*b_dd + dt*source(x_int, (n+1)*dt)
        u[n+1] = np.linalg.solve(LHS,RHS)
    return x_int,u[-1,:]



x,y = CrankNicholsonDirichlet(70,0,1,0,1.0,0.1,0.0,3.0,f,source)



plt.plot(x,y)
plt.show()

def CrankNicolson(N,domain_start,domain_end,D,t_final,dt,bc_type,bc_left,bc_right,initial_condition,source,*args):
    """
    A function that uses Crank-Nicolson method to approximate the solution to the diffusion equation

    SYNTAX 
    ------
    The function is called in the following way:

    CrankNicolson(N,domain_start,domain_end,D,t_final,dt,bc_type,bc_left,bc_right,initial_condition,source,*args)

    WHERE:

        N: 'N' is an integer value describing the number of points you'd like to discretise the domain into

        domain_start: 'domain start' 



    """

    # Checks for N
    if N < 5:
        raise ValueError("'N' must be greater than 5 to provide an accurate representation of the solution")
    if not isinstance(N, int):
        raise ValueError("'N' must be an integer value")
    if N > 200:
        raise RuntimeError("The size of 'N' leads to large computational compexity")
    
    # Checks for domain values

    if not isinstance(domain_start,int):
        raise ValueError("'domain_start' must be an integer")
    if domain_start > domain_end:
        raise ValueError("The starting point of the domain must be before the endpoint")
    if not isinstance(domain_end,int):
        raise ValueError("'domain_end' must be an integer value")
    
    # Checks for Diffusion coefficient

    if not isinstance(N,(float,int)):
        raise ValueError("'D' must be either a decimal or integer")
    if D < 0:
        raise ValueError("'D' must be a positive value")    
    
    # checks for t_final 

    if not isinstance(t_final,(float,int)):
        raise ValueError("'t_final' must be either a decimal or integer")
    if t_final < 0:
        raise ValueError("'t_final' must be a positive value")
    
    # checks for dt, need to check these with TA because very subjective dependent on the size of the time domain

    if not isinstance(dt,float):
        raise ValueError("'dt' must be a decimal value") 
    # Checking boundary conditions
    if not isinstance(bc_left,float):
        raise ValueError("'bc_left must be a decimal value")
    # Write seperate bc_right conditions for the different methods (robin)
    # initial condition check
    if initial_condition == 'none':
        def initial_condition(x,*args):
            return 0
    elif callable(initial_condition) == False:
        raise TypeError("'initial_condition' must be a callable function")
    # source checks
    if source == 'none':
        def source(x,*args):
            return 0
    elif callable(source) == False:
        raise TypeError("'source' must be a callable function")
    if bc_type == 'dirichlet':
        if not isinstance(bc_right,float):
            raise ValueError("'bc_right' must be a decimal value")
        x,y = CrankNicholsonDirichlet(N,domain_start,domain_end,D,t_final,dt,bc_left,bc_right,initial_condition,source,*args)
    elif bc_type == 'neumann':
        if not isinstance(bc_right,float):
            raise ValueError("'bc_right' must be a decimal value")
        x,y = CrankNicholsonNeumann(N,domain_start,domain_end,D,t_final,dt,bc_left,bc_right,initial_condition,source,*args)
    elif bc_type == 'robin':
        if not isinstance(bc_right,list):
            raise ValueError("'bc_right' must be a list containing both beta and gamma values")
        CrankNicholsonRobin(N,domain_start,domain_end,D,t_final,dt,bc_left,bc_right,initial_condition,source,*args)
    return x,y 





