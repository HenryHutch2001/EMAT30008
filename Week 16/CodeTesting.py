# %%
import scipy
from scipy.optimize import root 
import matplotlib.pyplot as plt
import numpy as np
from math import nan
# %%
def shooting(ode,U): #Finds the points and time period of the limit cycle
    x0,y0,T = U #Extracting the constituent parts of the input for the shooting function
    X0 = np.array((x0,y0)) #Putting the initial conditions into an array called X0, which is used in the 
    #ODE solver as y 
    t_eval = np.linspace(0,T,int(10*T)) #Defines the time for which we evaluate the ODE between
    condition_1 = X0 - scipy.integrate.solve_ivp(ode,[0, T],X0,args=[a,b,d], t_eval=t_eval, rtol = 1e-4).y[:,-1]
    #This line basically says that the first condition is where the initial guess, minus the solution for the time period T
    #is equal to 'condition 1' 
    condition_2 = ode(0,X0,a,b,d)[0] 
    #This is the initial phase condition for the limit cycle, which returns the value of dxdt at t=0 for the initial condition
    #although the equations do not depend on t. It effectively returns the gradient of the x line for which we want to find the roots
    # of using root. 
    return [*condition_1,condition_2] # THe * gets rid of the list within
