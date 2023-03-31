# %%
import scipy
from scipy.optimize import root 
import matplotlib.pyplot as plt
import numpy as np
from math import nan
import unittest
# %% Defining the ODE

def shooting(x0, ode, beta):
    """
    A function that uses the numerical shooting method in order to find the limit cycles of an ode

    SYNTAX
    ------
    The function is called the following way;

    shooting(x0, ode, beta):

        x0: the x0 input is a list containing an initial guess of the initial values of the limit cycle for the specified ode, the form of
        this input should be as follows;

            x0 = [x0,y0,T]

        ode: the ode input is the ordinary differential equation for which we want to find the limit cycle of, the form of the ode
        input should be as follows;

            def ode(t, y, beta): Defining the ode as a function with inputs t, y, and beta, time, initial conditions and parameter
            u1 = y[0]
            u2 = y[1]
            du1_dt = beta*u1 - u2 -u1*((u1**2) + (u2**2))
            du2_dt = u1 + beta*u2 -u2*((u1**2) + (u2**2))
            return [du1_dt, du2_dt] #Returns the value of the ode as an array

        beta: the beta input is the parameter for the specified ode

    OUTPUT
    ------
        The shooting function returns the correct initial values of the limit cycle
    """
    def cons(x0, ode, beta): #Finds the points and time period of the limit cycle
        condition_1 = x0[:2]- scipy.integrate.solve_ivp(lambda t, y: ode(t, y, beta), [0, x0[2]], x0[:2], rtol=1e-4).y[:,-1]
        condition_2 = ode(0, x0[:2], beta)[0] 
        return [*condition_1, condition_2]
    result = scipy.optimize.root(cons, x0=x0, args=(ode, beta))
    return result.x

def ode(t,y,beta):
    u1 = y[0]
    u2 = y[1]
    du1_dt = beta*u1 - u2 -u1*((u1**2) + (u2**2))
    du2_dt = u1 + beta*u2 -u2*((u1**2) + (u2**2))
    return [du1_dt, du2_dt]

def Hopf(y,t,beta):
    u1 = y[0]
    u2 = y[1]
    du1_dt = beta*u1 - u2 -u1*((u1**2) + (u2**2))
    du2_dt = u1 + beta*u2 -u2*((u1**2) + (u2**2))
    return [du1_dt, du2_dt]

x = shooting([1,1,20],ode,0)
# %%
x0 = [1,1,20]

def NaturalODE(ode1,ode2,x0,p0,p1,Steps):
    initial = shooting(x0,ode,p0)
    t = initial[2]
    initial_solution = initial[0:2]
    p_range = np.linspace(p0,p1,Steps)
    Solutions = np.array([initial_solution])
    p_values = np.array([p0])
    for i in range(1,len(p_range)):
        p = p_range[i]
        approx = Solutions[-1]
        true = root(ode2,x0 = approx,args=(t,p))
        if true.success == True:
            Solutions = np.vstack([Solutions,true.x])
            p_values = np.append(p_values,p)
    return p_values,Solutions

x,y = NaturalODE(ode,Hopf,x0,0,2,100)
print(y[:,1])

plt.plot(x,y[:,0])
plt.plot(x,y[:,1])
plt.show()
# %%
