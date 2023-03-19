# %%
import numpy as np
from scipy.optimize import root
import matplotlib.pyplot as plt

# %%

def function(v):
    x = v[1]
    c = v[0]
    return [x**3 -x + c,c]

a = np.array([[1,1],[2,2]])

b = np.array([1,2])[np.newaxis]

print(np.shape(a),np.shape(b))
a = np.concatenate((a,b),axis = 0)
print(a)
#%%

def CubicContStep(f,x0,p0,p1):
    p_range = np.linspace(p0,p1,100) #Defining initial P values
    solutions = np.array([x0]) #Creating array of all true solutions to the equation, including the initial guess
    p_value = np.array([p0]) #Adding the first p value to the pvalues array
    for i in range(0,len(p_range)-1): 
        p = p_range[i]
        predicted_value = solutions[-1]
        sol = root(f,x0=[p,predicted_value])
        if sol.success == True:
            solutions = np.append(solutions,sol.x[1])
            p_value = np.append(p_value,p)
    v0 = [p_value[1],solutions[1]] #Basically generating 2 values for which we know to be true as initial guesses 
    v1 = [p_value[2],solutions[2]]
    return v0,v1

v0,v1= CubicContStep(function,0,-2,2)
v0 = np.array(v0)
print(v0)

# %%
first = np.array([v0])

first = np.append(first,v1)
print(first)

# %%
def Conditions(v):
        Condition1 = function(v)
        Condition2 = np.dot((v-approx),secant)
        return np.array([Condition1[1],Condition2])

# %%

def PsuedoLength(f,x0,p0,p1):
    def Conditions(v):
        Condition1 = function(v)
        Condition2 = np.dot((v-approx),secant)
        return np.array([Condition1[1],Condition2])
    p_range = np.linspace(p0,p1,100)
    first,second = CubicContStep(f,x0,p0,p1)
    v0 = np.array([first])
    v1 = np.array([second])
    for i in range(0,len(p_range)-1):
         secant = v1[i]-v0[i]
         approx = v1[i]+secant
         solution = root(Conditions,x0 = approx)
         solution = solution.x


    return p_range,v1

PsuedoLength(function,0,-2,2)
# %%
approx = [-1.91919192,  0]
secant = [0.04040404, 0]

v0 = np.array([0,0])

def Conditions(v):
        Condition1 = function(v)
        Condition2 = np.dot((v-approx),secant)
        return np.array([Condition1[1],Condition2])

solution = root(Conditions,x0=approx)

solution = solution.x

print(solution)
print(v0)
v0 = np.concatenate(([v0],[solution]),axis = 0)

print(v0)
# %%

import numpy as np

a = np.array([1, 2, 3])
b = np.array([4, 5, 6])

c = np.concatenate(([a], [b]), axis=0)

print(c)

# %%
