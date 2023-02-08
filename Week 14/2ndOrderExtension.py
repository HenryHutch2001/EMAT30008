import numpy as np
import matplotlib.pyplot as plt

f1 = lambda t, x: np.array([x[1], -x[0]]) # x' = y, y' = -x
x0 = np.array([1, 0]) # Initial condition, x = 1, y = 0

def euler_step(f,xn,t,h):
    x = xn + h*f(t,xn)
    return x

def rk_step(f,xn,t,h):
    k1 = f(t,xn)
    k2 = f(t+h/2,xn + h*k1/2)
    k3 = f(t+h/2,xn+h*(k2/2))
    k4 = f(t+h,xn+(h*k3))
    x = xn + (h/6)*(k1 + 2*k2 + 2*k3 + k4)
    return x

def solve_to_system(f, x0, t1, t2, h, step_function):
    t = np.arange(t1,t2+h,h)
    x = x0
    xval = [x]
    for i in range(1,len(t)):
        x = step_function(f,x,t,h)
        xval.append(x)
    return t, xval
t,x = solve_to_system(f1,x0,0,10,0.01,rk_step)
plt.plot(t,x)
plt.show()