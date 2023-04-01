# %%
from My_Functions import shooting
import scipy
from scipy.optimize import root
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
# %%
def function(x,c):
    return x**3 -x+c
# %%
# %%
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
    return solutions[1:],p_value[1:]
x,y = CubicCont(function,-10,-2,2)
plt.plot(y,x,'o')
plt.show() 
#PYTEST
# %%
def CubicContStep(f,x0,p0,p1):
    p_range = np.linspace(p0,p1,10000)
    solutions = np.array([x0])
    p_value = np.array([p0])
    for i in range(0,len(p_range)-1):
        p = p_range[i]
        predicted_value = solutions[-1]
        sol = root(f,x0=predicted_value,args=(p,))
        if sol.success == True:
            solutions = np.append(solutions,sol.x)
            p_value = np.append(p_value,p)
    v0 = np.array([p_value[1],solutions[1]]) #Basically generating 2 values for which we know to be true as initial guesses 
    v1 = np.array([p_value[2],solutions[2]])
    return v0,v1
# %%
def PseudoLength(f,x0,p0,p1):
    def conditions(input):
        Condition1=function(input[0],input[1])
        Condition2=np.dot(((input)-approx),secant)
        return [Condition1,Condition2]
    first,second = CubicContStep(f,x0,p0,p1)
    solutions = np.array([first[1],second[1]])
    p_value = np.array([first[0],second[0]])
    v0 = first
    v1 = second
    secant = v1-v0
    approx = v1+secant
    p = p0
    while p >=p0 and p<=p1:
        secant = v1-v0
        approx = v1+secant
        sol = root(conditions,x0=approx,tol=1e-6)
        print(sol.x)
        v0 = v1
        v1 = np.array([sol.x[0],sol.x[1]])
        secant = v1-v0
        approx = v1+secant
        if sol.success == True:
            solutions = np.append(solutions,sol.x[1])
            p_value = np.append(p_value,sol.x[0])
            p = sol.x[0]
    return solutions,p_value
x,y= PseudoLength(function,-10,-2,2)
plt.plot(y,x,'o')
plt.show()
# %%
def ODEStep(ode,t,x0,p0,p1):
    p_range = np.linspace(p0,p1,1000)
    solutions = np.array([x0])
    p_value = np.array([p0])
    for i in range(0,len(p_range)-1):
        p = p_range[i]
        predicted_solution = solutions[-1]
        print(solutions)
        sol = root(ode,x0=predicted_solution,args=(t,p))
        if sol.success == True:
            solutions = np.append(solutions,sol.x)
            p_value = np.append(p_value,p)
        return solutions[1:],p_value[1:]
x,y = ODEStep(HopfNormal,1,-10,-3,3)
# %%
