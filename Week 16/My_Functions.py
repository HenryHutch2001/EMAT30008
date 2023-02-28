# %%
import numpy as np
import matplotlib.pyplot as plt
from sys import exit
import scipy
# %%
def euler_step(f,xn,t,h):
    f = np.array(f(t,xn))
    x = xn + h*f
    return x
# %%
def solve_toEU(f,x0,t1,t2,h):
   deltat_max = 1
   if h >= deltat_max:
    print("Step size too large for accurate approximation")
    exit(solve_toEU)
   t = np.arange(t1,t2+h,h)
   x = np.array([x0])
   for i in range(1,len(t)):
        Value = euler_step(f,x[i-1],t[i-1],h)
        x = np.vstack([x, Value]) 
   return t,x  
# %%
def rk_step(f,xn,t,h):
    k1 = np.array(f(t,xn))
    k2 = np.array(f(t+h/2,xn + h*k1/2))
    k3 = np.array(f(t+h/2,xn+h*(k2/2)))
    k4 = np.array(f(t+h,xn+(h*k3)))
    x = xn + (h/6)*(k1 + 2*k2 + 2*k3 + k4)
    return x
"Creating the function for approximation using RK4 method"
# %%
def solve_toRK(f,x0,t1,t2,h):
   deltat_max = 1
   if h >= deltat_max:
    print("Step size too large for accurate approximation")
    exit(solve_toRK)
   t = np.arange(t1,t2+h,h)
   x = np.array([x0])
   for i in range(1,len(t)):
        Value = rk_step(f,x[i-1],t[i-1],h)
        x = np.vstack([x,Value])
   return t,x   
# %%
def solve_to(f,x0,t1,t2,h):
    deltat_max = 1
    if h >= deltat_max:
        print("Step size too large for accurate approximation")
        exit(solve_to)
    method = input("Which approximation method would you like to use? Please enter either Euler or Runge-Kutta ").lower()
    if method == ("Euler").lower():
        solve_toEU(f,x0,t1,t2,h)
    elif method == ("Runge-Kutta").lower():
        solve_toRK(f,x0,t1,t2,h)
    else:
        print("Please provide a correct input")
        exit(solve_to)
# %%

def shooting(x0,ode):
    """
    A function that uses the numerical shooting method in order to find the limit cycles of an ode

    SYNTAX
    ------
    The function is called the following way;

    shooting(x0,ode):

        x0: the x0 input is a list containing an initial guess of the initial values of the limit cycle for the specified ode, the form of
        this input should be as follows;

            x0 = [x0,y0,T]

        ode: the ode input is the ordinary differential equation for which we want to find the limit cycle of, the form of the ode
        input should be as follows;

            def PredPrey(t,x0): Defining the ode as a function with inputs t and x0, time and initial conditions
            a = 1     |
            b = 0.1   |  Defining any parameters inside the function handle, you could leave them out of the function but you
            d = 0.1   |  would have to define them as arguments when calling the function
            x = x0[0] Splitting the initial conditions array into its constituent parts
            y = x0[1]
            dx_dt = x*(1-x) - (a*x*y)/(d+x) #Defining the differential equations
            dy_dt = b*y*(1-(y/x))
            return [dx_dt, dy_dt] #Returns the value of the ode as an array 
    
    OUTPUT
    ------
        The shooting function returns the correct initial values of the limit cycle
    """
    def cons(x0,ode): #Finds the points and time period of the limit cycle
        condition_1 = x0[:2]- scipy.integrate.solve_ivp(ode,[0, x0[2]],x0[:2], rtol = 1e-4).y[:,-1]
        condition_2 = ode(0,x0[:2])[0] 
        return [*condition_1,condition_2]
    result = scipy.optimize.root(cons, x0 = x0, args=(ode,))
    print(result.x)


# %%