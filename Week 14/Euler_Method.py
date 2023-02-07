import numpy as np
import matplotlib.pyplot as plt

f = lambda t, x:x

def exact_solution(t):
    return np.exp(t) 

"Creating the individual step function for Eulers method"
def euler_step(xn,t,h):
    x = xn + h*f(t,xn)
    "Single step Euler function"
    return x
"Creating the function for approximation using Eulers method"
def solve_toEU(x0,t1,t2,h):
   deltat_max = 1
   if h >= deltat_max:
    print("Step size too large for accurate approximation")
    exit(solve_toEU)
   t = np.arange(t1,t2+h,h)
   x = np.zeros_like(t)
   x[0] = x0
   for i in range(1,len(t)):
        x[i] = euler_step(x[i-1],t[i-1],h)
   return t,x   
"Creating the individual step function for RK4 method"
def rk_step(xn,t,h):
    k1 = f(t,xn)
    k2 = f(t+h/2,xn + h*k1/2)
    k3 = f(t+h/2,xn+h*(k2/2))
    k4 = f(t+h,xn+(h*k3))
    xn1 = xn + (h/6)*(k1 + 2*k2 + 2*k3 + k4)
    return xn1
"Creating the function for approximation using RK4 method"
def solve_toRK(x0,t1,t2,h):
   t = np.arange(t1,t2+h,h)
   x = np.zeros_like(t)
   x[0] = x0
   for i in range(1,len(t)):
        x[i] = rk_step(x[i-1],t[i-1],h)
   return t,x   

h = [0.1, 0.05, 0.01, 0.005, 0.001]
"Defining the different step sizes we use for calculation"
x0 = 1
"Defining the initial condition x0"
t1 = 0
"Defining the initial time, t = 0"
t2 = 1
"Defining the final time, t = 1"
fig, ax = plt.subplots()
"Setting up the axes for the plot"
EulerError = []
RK4Error =[]
for step in h:
    "For the number of steps in the array of different step sizes..."
    t, x = solve_toEU(x0, t1, t2, step)
    t, x1 = solve_toRK(x0, t1, t2, step )
    "return the x value and t value, aka the approximation for the function at each specific time step"
    y_exact = exact_solution(1)
    "Return the actual values of the function for each "
    EulerError.append(np.abs(y_exact-x[-1]))
    RK4Error.append(np.abs(y_exact-x1[-1]))
    "Return the error between our approximation and the value of the actual function for each timestep"
ax.loglog(h, EulerError, 'b-',label = 'Eulers method')
ax.loglog(h, RK4Error, 'r-', label = 'Runge-Kutta method')
ax.invert_xaxis()
plt.grid()
ax.set_xlabel("Step Size (h)")
ax.set_ylabel("Error")
ax.set_title("Error as a function of timestep for the approximation methods")
ax.legend()
plt.show()