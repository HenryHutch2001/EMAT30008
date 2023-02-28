# %%
import scipy
from scipy.optimize import root 
import matplotlib.pyplot as plt
import numpy as np
from math import nan
from scipy.signal import find_peaks
a = 1
b = 0.1
d = 0.1
# %%
def ode(t,y,a,b,d): #Keeping t in in case our ode requires it
    dx_dt = y[0]*(1-y[0]) - (a*y[0]*y[1])/(d+y[0]) #Defining the differential equation
    dy_dt = b*y[1]*(1-(y[1]/y[0]))
    return [dx_dt, dy_dt] #Returns the value of the ODE
# Solving the ode
t_eval = np.linspace(0,100,1000) #defines the timeframe for which we simulate the ode between
Solution = scipy.integrate.solve_ivp(ode,[0, 100],[0.5,0.5],t_eval=t_eval,args=[1,0.1,0.1],rtol = 1e-4)
#Uses the solve_ivp function to solve the ODE with the first argument being the function for which we solve
#The second being the time for which we solve between, the third being the initial conditions...
# %%
#Isolating a periodic orbit:
#For x:
# %%
#Plotting ODE
plt.plot(Solution.t,Solution.y[0,:], label = 'X') #Plotting X value by extracting the first column in the solution
plt.plot(Solution.t,Solution.y[1,:], label = 'y') #Plotting Y value by extracting the second column in the solution
plt.legend()
plt.show()
# %%
#Plotting the phase portrait
plt.plot(Solution.y[0,:],Solution.y[1,:]) #Plots the phase portrait, the value of X against Y
plt.show()
#We can see that the ODE eventually converges to a limit cycle where the values keep following the same cycle 
# %%
def shooting(x): #Finds the points and time period of the limit cycle
    condition_1 = x[:2]- scipy.integrate.solve_ivp(ode,[0, x[2]],x[:2],args=[a,b,d], rtol = 1e-4).y[:,-1]
    condition_2 = ode(0,x[:2],a,b,d)[0] 
    return [*condition_1,condition_2] # THe * gets rid of the list within
initial_guess = [0.8, 0.2,30]
result = scipy.optimize.root(shooting, x0 = initial_guess)
print(result.x)
