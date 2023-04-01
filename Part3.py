# %%
import numpy as np
import matplotlib.pyplot as plt

def ode(t,y): #Keeping t in in case our ode requires it
    dx_dt = y[1]
    dy_dt = -y[0]
    return [dx_dt, dy_dt]

def HopfNormal(t,y,beta=2):
    u1 = y[0]
    u2 = y[1]
    du1_dt = beta*u1 - u2 - u1*((u1**2) + (u2**2))
    du2_dt = u1 + beta*u2 - u2*((u1**2) + (u2**2))
    return [du1_dt, du2_dt]
# %%
def ode1(t,y):
    a = 1
    b = 0.1
    d = 0.1 #Keeping t in in case our ode requires it
    dx_dt = y[0]*(1-y[0]) - (a*y[0]*y[1])/(d+y[0]) #Defining the differential equation
    dy_dt = b*y[1]*(1-(y[1]/y[0]))
    return [dx_dt, dy_dt] #Returns the value of the ODE
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

t,x,y = solve_toEU(ode,[0,0],0,10,0.5)
print(x,y)
plt.plot(t,x)
plt.plot(t,y,'o')
plt.show()
# %%
