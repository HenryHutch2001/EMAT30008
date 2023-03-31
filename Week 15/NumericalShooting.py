# %%
import scipy
from scipy.optimize import root
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import numpy as np
from math import nan
from My_Functions import solve_toEU,solve_to,shooting
# %%
a = 1
b = 0.1
d = 0.1
def ode(t,y,a,b,d): #Keeping t in in case our ode requires it
    x = y[0]
    y = y[1]
    dx_dt = x*(1-x) - (a*x*y)/(d+x) #Defining the differential equation
    dy_dt = b*y*(1-(y/x))
    return [dx_dt, dy_dt] #Returns the value of the ODE


result = solve_ivp(ode,[0,10],[3,3],args=(1,0.1,0.1)).y
# Solving the ode
t_eval = np.linspace(0,100,1000) #defines the timeframe for which we simulate the ode between
t,x = solve_toEU(ode,[0.5,0.5],0,100,0.01,1,0.1,0.1)

#%%
#Uses the solve_ivp function to solve the ODE with the first argument being the function for which we solve
#The second being the time for which we solve between, the third being the initial conditions...
# %%

# %%
#Plotting the phase portrait
plt.plot(x[:,0],x[:,1]) #Plots the phase portrait, the value of X against Y
plt.show()
#We can see that the ODE eventually converges to a limit cycle where the values keep following the same cycle 
# %%
x = [30,0.8,0.2]
a = 1
b = 0.1
d = 0.1

y = shooting(x,ode,a,b,d)
print(y)

""" def shooting1(x,ode,*args):
    Condition1 = x[1:len(x)]-solve_ivp(ode,[0,x[0]],x[1:len(x)],args=([*args])).y[:,-1]
    Condition2 = ode(0,x[1:len(x)],*args)[0]
    return [*Condition1,Condition2]
Result = root(shooting1,x0 = x,args=(ode,a,b,d))
print(Result.x) """

