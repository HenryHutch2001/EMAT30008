#Numerical shooting 
import numpy as np
from matplotlib import pyplot as plt
from scipy import integrate


def ode(u,t):
    a = 1
    b = 0.1
    d = 0.1
    x = u[0]
    y = u[1]
    dxdt = x*(1-x) -(a*x*y)/(d+x)
    dydt = (b*y)*(1 - y/x)
    return [dxdt, dydt]
u0 = [5,5]
t = np.linspace(0,200,200)
x = integrate.odeint(ode,u0,t)
print(x[:,0])
print(len(t))

plt.plot(t,x[:,0])
plt.plot(t,x[:,1])
plt.show()


