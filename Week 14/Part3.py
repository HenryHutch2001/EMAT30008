# %%
import numpy as np
import matplotlib.pyplot as plt

def ode(t,y): #Keeping t in in case our ode requires it
    dx_dt = y[1]
    dy_dt = -y[0]
    return dx_dt, dy_dt
# %%
def ode1(t,y):
    a = 1
    b = 0.1
    d = 0.1 #Keeping t in in case our ode requires it
    dx_dt = y[0]*(1-y[0]) - (a*y[0]*y[1])/(d+y[0]) #Defining the differential equation
    dy_dt = b*y[1]*(1-(y[1]/y[0]))
    return dx_dt, dy_dt #Returns the value of the ODE
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
   print(x)  
   return t,x  

# %%
t,x = solve_toEU(ode,[1,1],0,100,0.01)
plt.plot(t,x)
