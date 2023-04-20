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
    return np.zeros(np.size(x))

def source(x,t,u):
    return np.ones(np.size(x))


def CrankNicholsonDirichlet(N,domain_start,domain_end,D,t_final,dt,bc_left,bc_right,initial_condition,source,*args):  
    a = domain_start
    b = domain_end
    A_dd,b_dd,x_int,dx = CreateAandb(N,a,b,'dirichlet',bc_left,bc_right)
    C = dt*D/(dx**2)
    N_t = ceil(t_final/dt) 
    u = np.zeros((N_t+1, N-1))
    u[0,:] = initial_condition(x_int)
    LHS = ((np.identity(N-1))-((C/2)*A_dd))
    for n in range(0,N_t):
        RHS = ((np.identity(N-1))+((C/2)*A_dd))@u[n] + C*b_dd + dt*source(x_int, (n+1)*dt,u[n],*args)
        u[n+1] = np.linalg.solve(LHS,RHS)
    return x_int,u[-1,:]

def CrankNicholsonNeumann(N,domain_start,domain_end,D,t_final,dt,bc_left,bc_right,initial_condition,source,*args):
    a = domain_start
    b = domain_end
    A_dd,b_dd,x_int,dx = CreateAandb(N,a,b,'neumann',bc_left,bc_right)
    C = dt*D/(dx**2)
    N_t = ceil(t_final/dt) 
    u = np.zeros((N_t+1, N))
    u[0,:] = initial_condition(x_int)
    LHS = ((np.identity(N))-((C/2)*A_dd))
    for n in range(0,N_t):
        RHS = ((np.identity(N))+((C/2)*A_dd))@u[n] + C*b_dd + dt*source(x_int, (n+1)*dt,u[n],*args)
        u[n+1] = np.linalg.solve(LHS,RHS)
    return x_int,u[-1,:]

def CrankNicholsonRobin(N,domain_start,domain_end,D,t_final,dt,bc_left,bc_right,initial_condition,source,*args):
    a = domain_start
    b = domain_end
    A_dd,b_dd,x_int,dx = CreateAandb(N,a,b,'robin',bc_left,bc_right)
    C = dt*D/(dx**2)
    N_t = ceil(t_final/dt) 
    u = np.zeros((N_t+1, N))
    u[0,:] = initial_condition(x_int)
    LHS = ((np.identity(N))-((C/2)*A_dd))
    for n in range(0,N_t):
        RHS = ((np.identity(N))+((C/2)*A_dd))@u[n] + C*b_dd + dt*source(x_int, (n+1)*dt,u[n],*args)
        u[n+1] = np.linalg.solve(LHS,RHS)
    return x_int,u[-1,:]

def CrankNicolson(N,domain_start,domain_end,D,t_final,dt,bc_type,bc_left,bc_right,initial_condition,source,*args):
    """
    A function that uses Crank-Nicolson method to approximate the solution to the diffusion equation

    SYNTAX 
    ------
    The function is called in the following way:

    CrankNicolson(N,domain_start,domain_end,D,t_final,dt,bc_type,bc_left,bc_right,initial_condition,source,*args)

    WHERE:

        N: 'N' is an integer value describing the number of points you'd like to discretise the domain into. The form of this input should be as follows;

            N = 1

        domain_start/domain_end: 'domain_start' and 'domain_end are integer values describing the initial and final values in the spatial domain for which the solution is approximated for. The form of this input should be as follows;

            domain_start = 0
            domain_end = 1

        D: 'D' is an integer/floating point value describing the diffusion coefficient of the diffusion equation. 
        The form of this input should be as follows;

            D = 0.5

        t_final: 't_final' is an integer/floating point number defining the time that the approximation solves up to, starting at zero. 
        The form of this input should be as follows;

            t_final = 1

        dt: 'dt' is a decimal value defining the time step size, the time domain is split into sections equal 
        to the size of dt and the solition is calculated at each point. The form of this input should be as follows;

            dt = 0.1

        bc_left: 'bc_left' is a decimal value defining the left boundary condition. f(domain_start,t) = bc_left.  
        The form of this input should be as follows;

            bc_left = 0.0

        bc_left: 'bc_right' is a decimal value defining the right boundary condition. f(domain_end,t) = bc_right.  
        The form of this input should be as follows;

            bc_right = 1.0

        bc_type: 'bc_type' is a string describing the type of boundary conditions imposed on the equation, 
        inputs are either 'dirichlet', 'neumann' or 'robin'. The form of this input should be as follows;

            bc_type = 'neumann'

        initial_condition: 'initial_condition' is a function imposing an initial condition on the solution, 
        if no initial condition is wanted, input 'none' and the function will remove any condition. 
        The form of this input should be as follows;

            def f(x,*args):
                return np.ones(size(x))

        source: 'source' is a function describing the source term 'q(x,t,u;parameters)' 

            def source(x,t,u):
                return np.sin(math.pi*x)
        
    OUTPUT
    ------

    Returns 2 numpy arrays of equal lengths, with the first array containing the interior 
    gridpoints and the second containing the solutions for these gridpoints.

    """
    if N < 5:
        raise ValueError("'N' must be greater than 5 to provide an accurate representation of the solution")
    if not isinstance(N, int):
        raise ValueError("'N' must be an integer value")
    if N > 200:
        raise RuntimeError("The size of 'N' leads to large computational compexity")
    if not isinstance(domain_start,int):
        raise ValueError("'domain_start' must be an integer")
    if domain_start > domain_end:
        raise ValueError("The starting point of the domain must be before the endpoint")
    if not isinstance(domain_end,int):
        raise ValueError("'domain_end' must be an integer value")
    if not isinstance(N,(float,int)):
        raise ValueError("'D' must be either a decimal or integer")
    if D < 0:
        raise ValueError("'D' must be a positive value")    
    if not isinstance(t_final,(float,int)):
        raise ValueError("'t_final' must be either a decimal or integer")
    if t_final < 0:
        raise ValueError("'t_final' must be a positive value")
    if not isinstance(dt,float):
        raise ValueError("'dt' must be a decimal value") 
    if not isinstance(bc_left,float):
        raise ValueError("'bc_left must be a decimal value")
    if initial_condition == 'none':
        def initial_condition(x,*args):
            return 0
    elif callable(initial_condition) == False:
        raise TypeError("'initial_condition' must be a callable function")
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
    else:
        raise ValueError("'bc_type' must be either 'dirichlet', 'neumann' or 'robin'")
    return x,y 


x,y = CrankNicolson(100,0,1,1,2,0.1,'dirichlet',0.0,0.0,f,'none')
print(type(y))
print(type(x))
print(math.exp(-0.2*math.pi**2))
plt.plot(x,y)
plt.show()

