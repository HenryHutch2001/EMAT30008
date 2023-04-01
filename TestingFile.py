# %%
from My_Functions import shooting,solve_to
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import root
import matplotlib.pyplot as plt
# %% Defining the Hopf Normal Form equation in Cartesian and Polar coordinates 

def HopfNormalCartesian(t,y,beta): # Cartesian
    u1,u2 = y
    du1_dt = beta*u1 - u2 - u1*(u1**2 + u2**2)
    du2_dt = u1 + beta*u2 - u2*(u1**2 + u2**2)
    return [du1_dt,du2_dt]

def HopfNormalPolar(t,y,beta): # Polar 
    r,theta = y 
    drdt = r * (beta - r**2 - 1)
    dthetadt = 1
    return [drdt, dthetadt]

# %% Determining initial starting conditions using shooting for both forms of the ODE
CartesianSolution = shooting([20,1,1],HopfNormalCartesian,2)
PolarSolution = shooting([20,1,1],HopfNormalPolar,2)

print(CartesianSolution,PolarSolution)

# %%

Result = solve_ivp(HopfNormalPolar,[0,20],[1,1],args=(2,))

plt.plot(Result.t,Result.y[0,:])
plt.plot(Result.t,Result.y[1,:])
plt.show()

# %%
t,x = solve_to(HopfNormalPolar,[1,1],0,20,0.001,2)
plt.plot(t,x)
plt.show()

# %% Attempting Natural Parameter Continuation for the Polar form:

def NaturalParamODE(ode,x0,p0,p1):
    # Determining initial true solution using shooting
    x = shooting(x0,ode,p0)
    t = x[0]
    x = x[1:len(x)]
    p_range = np.linspace(p0,p1,1000)
    Solutions = np.array([x])
    ParameterValues = np.array([p0])
    for i in range(0,len(p_range)-1):
        p = p_range[i]
        Prediction = Solutions[-1]
        sol = root(ode,x0=Prediction,args=(Prediction,p))
        if sol.success == True:
            Solutions = np.vstack([Solutions,sol.x])
            ParameterValues = np.append(ParameterValues,p)
    return Solutions,ParameterValues

x,p = NaturalParamODE(HopfNormalPolar,[30,0,1],0,2)
print(x)
# %%
