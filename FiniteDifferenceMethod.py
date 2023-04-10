# %%
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import root
import math
def FiniteDifferenceRobin(source,D,N,alpha,betagamma,*args):
    a = 0 
    b = 1
    GridSpace = np.linspace(a,b,N)
    Initial = 0.1 * GridSpace
    dx = (b-a)/N
    def SolveSource(u):
        F = np.zeros(N)
        F[0] = D*((u[1]-2*u[0]+alpha)/(dx**2))+source(GridSpace[0],F[0],*args)
        for i in range(1,N-1):
            F[i] = D*()       
def FiniteDifferenceNeumann(source,D,N,alpha,beta,*args):
    a = 0
    b = 1
    GridSpace = np.linspace(a,b,N)
    Initial = 0.1 * GridSpace
    dx = (b-a)/N
    def SolveSource(u):
        F = np.zeros(N)
        F[0] = D*((u[1]-2*u[0]+alpha)/(dx**2))+source(GridSpace[0],F[0],*args)
        for i in range(1,N-1):
            F[i] = D*((u[i+1]-2*u[i]+u[i-1])/(dx**2))+source(GridSpace[i],F[i],*args)
        F[N-1] = D*((-2*(u[N-1])+2*u[N-2])/(dx**2)+((2*beta)/dx) +source(GridSpace[N-1],F[N-1],*args))
        return F
    Solution = root(SolveSource,x0 = Initial)
    return GridSpace,Solution.x



def FiniteDifferenceDirichlet(source,D,N,alpha,beta,*args): #Perfect
    a = 0
    b = 1
    GridSpace = np.linspace(a,b,N+1)
    dx = (b-a)/N 
    Interior = GridSpace[1:-1]
    Guess = Interior * 0.1
    def SolveSource(u):
        F = np.zeros(N-1)
        F[0] = D*((u[1]-2*u[0]+alpha)/(dx**2))+source(Interior[0],F[0],*args)
        for i in range(1,N-2):
            F[i] = D*((u[i+1]-2*u[i]+u[i-1])/(dx**2))+source(Interior[i],F[i],*args)
        F[N-2] = D*((beta-2*u[N-2]+u[N-3])/(dx**2))+source(Interior[N-2],F[N-2],*args)
        return F
    Solution = root(SolveSource,x0 = Guess) 
    return Interior,Solution.x

def source(x,f,beta):
    return x*4 + beta

x,y = FiniteDifferenceDirichlet(source,1,20,0.0,1.0,1)
plt.plot(x,y)
plt.show()


# %%
def PDE(x):
    return 0

def q(x):
    return x**2
# %%
a = 0 #Defining the problem parameters
b = 1
MU = 4
alpha = 0.0 #Defining the boundary conditions 
beta = 1.0

N = 20 #Defining the number of gridpoints

GridSpace = np.linspace(a,b,N-1) #Defining the gridspace
dx = (b-a)/N 
Initial = 0.1 * GridSpace

# %%
def FiniteSolverNoSource(u,N,dx,alpha,beta):
    F = np.zeros(N-1)
    F[0] = ((u[1]-2*u[0]+alpha)/(dx**2))
    for i in range(1,N-2):
        F[i] = ((u[i+1]-2*u[i]+u[i-1])/(dx**2))
    F[N-2] = ((beta-2*u[N-2]+u[N-3])/(dx**2))
    return F


def True1(GridSpace,a,b,alpha,beta):
    return (((beta-alpha)/(b-a))*(GridSpace-a))+alpha
x = True1(GridSpace,a,b,alpha,beta)
Solution = root(FiniteSolverNoSource,x0 = Initial,args=(N,dx,alpha,beta))   
plt.plot(GridSpace,Solution.x)
plt.plot(GridSpace,x)
plt.show()
# %%
def FiniteSolverSimpleSource(u,N,dx,alpha,beta,D=1,q=1):
    F = np.zeros(N-1)
    F[0] = D*((u[1]-2*u[0]+alpha)/(dx**2))+q
    for i in range(1,N-2):
        F[i] = D*((u[i+1]-2*u[i]+u[i-1])/(dx**2))+q
    F[N-2] = D*((beta-2*u[N-2]+u[N-3])/(dx**2))+q
    return F

def True2(GridSpace,a,b,alpha,beta,D):
    return (-(1/2*D)*(GridSpace-a)*(GridSpace-b))+(((beta-alpha)/(b-a))*(GridSpace-a))+alpha
x = True2(GridSpace,a,b,alpha,beta,D=1)
plt.plot(GridSpace,x)
Solution = root(FiniteSolverSimpleSource,x0=Initial,args=(N,dx,alpha,beta))   
plt.plot(GridSpace,Solution.x)
plt.show()
# %%
def FiniteSolverDependentSource(u,N,dx,alpha,beta,D=1):
    F = np.zeros(N-1)
    F[0] = D*((u[1]-2*u[0]+alpha)/(dx**2))+q(GridSpace[0])
    for i in range(1,N-2):
        F[i] = D*((u[i+1]-2*u[i]+u[i-1])/(dx**2))+q(GridSpace[i])
    F[N-2] = D*((beta-2*u[N-2]+u[N-3])/(dx**2))+q(GridSpace[N-2])
    return F
Solution = root(FiniteSolverDependentSource,x0=Initial,args=(N,dx,alpha,beta))   
plt.plot(GridSpace,Solution.x)
plt.show()
# %%
def FiniteSolverDependentSourceU(u,MU,N,dx,alpha,beta,D=1):
    def q(x,f):
        return math.exp(MU*f)
    F = np.zeros(N-1)
    F[0] = D*((u[1]-2*u[0]+alpha)/(dx**2))+q(GridSpace[0],F[0])
    for i in range(1,N-2):
        F[i] = D*((u[i+1]-2*u[i]+u[i-1])/(dx**2))+q(GridSpace[i],F[i])
    F[N-2] = D*((beta-2*u[N-2]+u[N-3])/(dx**2))+q(GridSpace[N-2],F[N-2])
    return F
Solution = root(FiniteSolverDependentSourceU,x0=Initial,args=(MU,N,dx,alpha,beta))   
plt.plot(GridSpace,Solution.x)
plt.show()
# %%
