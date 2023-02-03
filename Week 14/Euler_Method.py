import numpy as np
import matplotlib.pyplot as plt

h = 0.1
"Defining the step size"

f = lambda t, x:x

def euler_step(xn,t,h):
    xn1 = xn + h*f(t,xn)
    print(xn1)

euler_step(1,0,h)