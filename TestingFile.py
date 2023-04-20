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
bc_right = 0
GridSpace = np.linspace(a,b,N+1)
x_ints = GridSpace[1:-1]


def q_func(t,x,u):
    return 0

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

A_dd,b_dd,dx=CreateAandbDirichlet(N,a,b,bc_left,bc_right)

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
# %%
# Function for solving with RK:
def RKSolver(N,domain_start,domain_end,D,bc_type,bc_left,bc_right,initial_condition,source,*args):
    a = domain_start
    b = domain_end
    if bc_type == 'dirichlet':
        A_dd,b_dd,x_int,dx = CreateAandb(N,a,b,'dirichlet',bc_left,bc_right)
    elif bc_type == 'neumann':
        A_dd,b_dd,x_int,dx = CreateAandb(N,a,b,'neumann',bc_left,bc_right)
    elif bc_type == 'robin':
        A_dd,b_dd,x_int,dx = CreateAandb(N,a,b,'robin',bc_left,bc_right)
    def system(t,u,source,*args):
        du_dt = (D/(dx**2))*(A_dd*u-b_dd-(dx**2)*source(t,x_int,u,*args))
        return du_dt
    t,u = solve_toRK(system,initial_condition(x_int),0,10,0.01,source,*args)
    return x_int,u[0,:],t

x,y,t = RKSolver(50,0,1,1,'dirichlet',0.0,0.0,IC,q_func)

plt.plot(x,y,'.')
plt.show()
# %%

def ExplicitEuler(N,domain_start,domain_end,D,t_final,dt,bc_type,bc_left,bc_right,initial_condition,source,*args):
    a = domain_start
    b = domain_end

    N_t = ceil(t_final/dt)
    t = dt * np.arange(N_t)
    if bc_type == 'dirichlet':
        A_dd,b_dd,x_int,dx = CreateAandb(N,a,b,'dirichlet',bc_left,bc_right)
    elif bc_type == 'neumann':
        A_dd,b_dd,x_int,dx = CreateAandb(N,a,b,'neumann',bc_left,bc_right)
    elif bc_type == 'robin':
        A_dd,b_dd,x_int,dx = CreateAandb(N,a,b,'robin',bc_left,bc_right)
    C = dt*D/(dx**2)
    u = np.zeros((N_t+1,N-1))
    u[0,:] = initial_condition(x_int)
    print(u)

    for n in range(0,N_t):
        u[n+1] = u[n] + C*(np.dot(A_dd,u[n])+ b_dd) + source(x_int,t[n],u[n],*args)
    return x_int,u[-1,:]

def q_func(t,x,u):
    return np.zeros(np.size(x))

def IC(x):
    return np.sin((np.pi*(x-a))/(b-a))

x,y = ExplicitEuler(60,a,b,D,1,0.1,'dirichlet',0.0,0.0,IC,q_func)
plt.plot(x,y,label='Approximation')
print(y)

x = np.linspace(a,b,1000)
t = 0

# Calculate the true solution at each (x,t) pair
def true_solution(x, t):
    return np.exp(-(D*np.pi**2*t)/(b-a)**2) * np.sin((np.pi*(x-a))/(b-a))

u_true = true_solution(x, t)
plt.plot(x, u_true,label='True Solution') 
plt.legend()
plt.show()
# %%
import numpy as np
from scipy.integrate import solve_bvp

def bratu(x, y, lambda_val):
    return np.vstack((y[1], -lambda_val*np.exp(y[0])))

def bc(ya, yb,p):
    return np.array([ya[0], yb[0]])

x = np.linspace(0, 1, 5)
y = np.zeros((2, x.size))
lambda_val = 2.0
sol = solve_bvp(bratu, bc, x, y, p=[lambda_val])

plt.plot(x,sol.y[0])
plt.show()
# %%
