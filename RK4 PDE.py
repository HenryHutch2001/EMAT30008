# %%
from scipy.integrate import solve_ivp
import math
from math import ceil
import numpy as np 
import matplotlib.pyplot as plt
import matplotlib.animation as animation 
from My_Functions import solve_toRK

N = 20
a = 0
b = 1
D = 1
GridSpace = np.linspace(a,b,N)
bc_left = 0
bc_right = 0

def CreateAandb(N,a,b, bc_left, bc_right):
    A = np.zeros((N,N))
    dx = (b-a)/N
    for i in range(1,N-1):
        A[i,i] = -2/((dx)**2)
        A[i,i+1] = A[i,i-1] = 1/((dx)**2)
    A[0,:] = [1] + [0]*(N-1)
    A[N-1,:] = [0]*(N-1) + [1]
    b = np.zeros(N)
    b[0] = bc_left
    b[N-1] = bc_right
    return A,b, dx

A_dd,b_dd,dx = CreateAandb(N,a,b,bc_left,bc_right)

def IC(x):
    return np.sin((np.pi*(x-a))/(b-a))

def system(t,u):
    du_dt = (D/(dx**2))*(A_dd*u+b_dd+(dx**2))
    return du_dt

def q_func(x, t, u):
    q = np.zeros_like(x) # initialize q as an array of zeros
    for i in range(len(x)):
        q[i] = x[i]**2 * t # modify q as a function of x, t, and u
    return q

def systemsource(t,u,q):
    q = q_func(GridSpace, t, u)
    du_dt = (D/(dx**2))*(A_dd*u+b_dd+(dx**2)*q)
    return du_dt


x0 = IC(GridSpace)
x = np.linspace(a,b,1000)
t = 0
def true_solution(x, t):
    return np.exp(-(D*math.pi**2*t)/(b-a)**2) * np.sin((math.pi*(x-a))/(b-a))

u_true = true_solution(x, t)

# Plot the true solution
plt.plot(x, u_true, label='True solution') 

t,x = solve_toRK(system,IC(GridSpace),0,10,0.1)
# extract the solution values for each time step
plt.plot(GridSpace,x[0,:],'o')
plt.xlabel('x')
plt.ylabel('u(x)')
plt.show()

t1,x1 = solve_toRK(systemsource,IC(GridSpace),0,10,0.1,q_func)

plt.plot(GridSpace,x1[0,:])
plt.show()






# %%
