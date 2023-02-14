#Numerical shooting 
import numpy as np
from matplotlib import pyplot as plt
from scipy import integrate


def ode(u,t): #Defining the ODE
    a = 1
    b = 0.4
    d = 0.1
    x = u[0]
    y = u[1]
    dxdt = x*(1-x) -(a*x*y)/(d+x)
    dydt = (b*y)*(1 - y/x)
    return [dxdt, dydt]

u0 = [5,5] #Initial conditions, e.g number of prey/predators
t = np.linspace(0,200,200) #Defining the time 
x = integrate.odeint(ode,u0,t) #Using the integrate.ode function to solve the ODE

plt.plot(t,x[:,0],label = "Predators") #Plotting the first column of the solution 
plt.plot(t,x[:,1], label = "Prey") #Plotting the second column
plt.xlabel('t')
plt.ylabel('Population') #Graph stuff
plt.legend()
plt.show()


