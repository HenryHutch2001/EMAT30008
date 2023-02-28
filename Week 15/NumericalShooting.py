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
#Determining the appropriate phase conditions for a limit cycle: 
    #A limit cycle is a closed trajectory in phase space, lets say the initial x0 = 0.8, by looking 
    #at the graph we can see that the value will eventually converge.
    #So our conditions for our limit cycle are x(0)=0.8 (as an initial guess), as the x value in
    #the ODE will return to this value after a period

#To isolate a periodic orbit i'd just 

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
initial_guess = [0.8, 0.2,30] #Defines our initial guess for the shooting method, which says we guess
#that the conditions 1&2 are satisfied at the point x = 0.8, y = 0.2, with a period of 30 seconds

result = scipy.optimize.root(shooting, x0 = initial_guess)#Finds the parameters for x0,y0 and T for which both conditions
#are satisfied.
print(result.x) #Prints the values for which the limit cycle exists.
# a.k.a the x line returns to a peak of 0.81908554 after the period T = 34.070..., and the y line returns to the value 0.166 after the same period



















# %%
