
# %%
import numpy as np
import matplotlib.pyplot as plt
from math import ceil
from My_Functions import solve_toRK,solve_toEU
# %%
N = 100
a = 0
b = 1
D = 1
bc_left = 0
bc_right = [1,0]
GridSpace = np.linspace(a,b,N+1)
x_ints = GridSpace[1:]


def q_func(t,x,u):
    return np.ones(np.size(x))

def IC(x):
    return np.sin((np.pi*(x-a))/(b-a))

def CreateAandbDirichlet(N,a,b,bc_left,bc_right):
    A = np.zeros((N-1,N-1))
    dx = (b-a)/N
    for i in range(0,N-1):
        A[i,i] = -2
    for i in range(0,N-2):
        A[i,i+1] = A[i,i-1] = 1
    B = np.zeros(N-1)
    B[0] = bc_left
    B[N-2] = bc_right
    return A,B.T,dx

def CreateAandbNeumann(N,a,b,bc_left,bc_right):
    A = np.zeros((N,N))
    dx = (b-a)/N
    for i in range(0,N):
        A[i,i] = -2
    for i in range(0,N-1):
        A[i,i+1] = A[i,i-1] = 1
    A[N-1,N-2] = 1
    B = np.zeros(N)
    B[0] = bc_left
    B[N-1] = 2*bc_right*dx
    return A,B.T,dx

def CreateAandbRobin(N,a,b,bc_left,bc_right):
    A = np.zeros((N,N))
    dx = (b-a)/N
    beta,gamma = bc_right
    for i in range(0,N-1):
        A[i,i] = -2
    A[N-1,N-1] = -2*(1+gamma*dx)
    for i in range(0,N-1):
        A[i,i+1] = A[i,i-1] = 1
    A[N-1,N-2] = 2
    B = np.zeros(N)
    B[0] = bc_left
    B[N-1] = 2*beta*dx
    return A,B.T,dx

A_dd,b_dd,dx=CreateAandbRobin(N,a,b,bc_left,bc_right)

def system(t,u,q,*args):
    du_dt = (D/(dx**2))*(A_dd*u-b_dd-(dx**2)*q(x_ints,*args))
    return du_dt

t,x = solve_toRK(system,IC(x_ints),0,10,0.1,q_func,0,x_ints)
plt.plot(x_ints,x[0,:])
plt.show()
# %%

def CreateAandb(N,a,b,bc_type,bc_left,bc_right):
    GridSpace = np.linspace(a,b,N+1)
    if bc_type == 'dirichlet':
        x_ints = GridSpace[1:-1]
        A_dd,b_dd,dx=CreateAandbDirichlet(N,a,b,bc_left,bc_right)
    elif bc_type == 'neumann':
        x_ints = GridSpace[1:]
        A_dd,b_dd,dx=CreateAandbNeumann(N,a,b,bc_left,bc_right)
    elif bc_type == 'robin':
        x_ints = GridSpace[1:]
        A_dd,b_dd,dx=CreateAandbNeumann(N,a,b,bc_left,bc_right)
    return A_dd,b_dd,x_ints,dx
