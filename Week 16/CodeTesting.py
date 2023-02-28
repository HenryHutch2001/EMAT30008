# %%
import scipy
from scipy.optimize import root 
import matplotlib.pyplot as plt
import numpy as np
from math import nan
from My_Functions import shooting
# %%

def ode(t,y):
    sigma = -1
    beta = 1
    u1 = y[0]
    u2 = y[1]
    du1_dt = beta*u1 - u2 + sigma*u1*(u1**2 + u2**2)
    du2_dt = u1 + beta*u2 +sigma*u2*(u1**2 + u2**2)
    return [du1_dt, du2_dt]

shooting([0.5,0.4,20],ode)

# %%
