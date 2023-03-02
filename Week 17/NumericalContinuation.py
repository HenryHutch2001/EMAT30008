from My_Functions import shooting
import scipy
from scipy.optimize import root
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

def function(x,c):
    return x**3 -x + c

def CubicCont(f,x0,p0,p1):
    p_range = np.linspace(p0,p1,100)
    solutions = np.array([x0])
    for i in range(0,len(p_range)-1):
        p = p_range[i]
        x = solutions[i]
        predicted_value = f(x,p)
        sol = root(f,x0=predicted_value,args=(p))
        solutions = np.append(solutions,sol.x)
    return p_range,solutions

x,y = CubicCont(function,0,-2,2)
plt.plot(x,y)
plt.show()