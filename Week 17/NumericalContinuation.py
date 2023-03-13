# %%
from My_Functions import shooting
import scipy
from scipy.optimize import root
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
# %%
def function(x,c):
    return x**3 -x + c

v = np.array(([0, 1],[1, 2]))
a0,x0 = v[0]
a1,x1 = v[1]
secant = v[0]-v[1]
v_predict = v[1]+secant
v_true = root(function,x0=x1,args=(a1))
print(v_true.x)
# %%
def HopfNormal(t,y,beta):
    sigma = -1
    u1 = y[0]
    u2 = y[1]
    du1_dt = beta*u1 - u2 + sigma*u1*((u1**2) + (u2**2))
    du2_dt = u1 + beta*u2 + sigma*u2*((u1**2) + (u2**2))
    return [du1_dt, du2_dt]

# %%
def CubicContStep(f,x0,p0,p1):
    p_range = np.linspace(p0,p1,1)
    solutions = np.array([x0])
    p_value = np.array([p0])

#Need to define a function for cubic continuation that generates 1 step 

def CubicCont(f,x0,p0,p1):
    p_range = np.linspace(p0,p1,1000)
    solutions = np.array([x0])
    p_value = np.array([p0])
    for i in range(0,len(p_range)-1):
        p = p_range[i]
        predicted_value = solutions[-1]
        sol = root(f,x0=predicted_value,args=(p))
        if sol.success == True:
            solutions = np.append(solutions,sol.x)
            p_value = np.append(p_value,p)
    return p_value[1:],solutions[1:]

x,y = CubicCont(function,1,-2,2)
plt.plot(x,y,'o')
plt.show() 

#Develop CubicContinuation so that it returns only 2 values, v0 and v1

#PYTEST
# %%

def CubePsuedo(f,x0,p0,p1):
    p_range = np.linspace(p0,p1,1000)
    boop = root(f,x0=x0,)
    v0 = 
# %%
""" def NumCont(ode,p0,p1,u0):
    p_range = np.linspace(p0,p1,100)
    solutions = np.array([u0])
    for i in range(0,len(p_range)-1):
        p = p_range[i]
        x = solutions[i]
        y0 = ode(x,p)
        sol = solve_ivp(ode,[0,1],y0,args=(p,))
        solutions = np.append(solutions,sol.y)
    return p_range,solutions

x,y = NumCont(function,-2,2,1)
 """
# %%

""" p_range = np.linspace(0,2,100)
#Vary the beta parameter between 0 and 2 
x0 = [1,1]
#Initial conditions for the ODE 
t = 1
solutions = np.array([x0])

for i in range(0,len(p_range)-1):
    y0 = HopfNormal(t,x0,p_range[i])
    sol = solve_ivp(HopfNormal,t_span=[0,1],y0=y0,args=(p_range[i],))
    solutions = np.append(solutions,sol.y)
print(len(solutions)) """
