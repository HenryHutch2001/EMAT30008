# %%
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve

def function(x,c):
    return x**3 -x+c

def NumCont(f, x0, p0, p1):
    p_range = np.linspace(p0, p1, 1000)
    solutions = np.array([x0])
    p_value = np.array([p0])
    for i in range(0, len(p_range) - 1):
        p = p_range[i]
        predicted_value = solutions[-1]
        sol = fsolve(f, x0=predicted_value, args=(p,), fprime=None)
        if np.all(np.isfinite(sol)):
            solutions = np.append(solutions, sol)
            p_value = np.append(p_value, p)
    return p_value[1:], solutions[1:]

x,y = NumCont(function,1,-2,2)
plt.plot(x,y,'.')
print(y)
plt.show()

# %%
def PseudoLength(f,x0,p0,p1):
    def conditions(input):
        Condition1=function(input[0],input[1])
        Condition2=np.dot(((input)-approx),secant)
        return [Condition1,Condition2]
    x1,y1 = NumCont(f,x0,p0,p1)
    first = [x1[0],y1[0]]
    second = [x1[1],y1[1]]
    # Generate secant 
    
x,y= PseudoLength(function,-1,-2,2)
plt.plot(y,x,'o')
plt.show()

# %%
