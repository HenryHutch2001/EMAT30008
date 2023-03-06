# %%
import numpy as np
import matplotlib as plt
import My_Functions as mf
import scipy
from scipy.optimize import root 
# %%
c = -2
#rearranging the equation for x:
def paramcont(x,c):
    x1 = []
    c1 = []
    for i in range(0,len(c)):
        x1.append(x(i-1)**3 + c(i-1))
        c1.append(x(i)**3 - x(i))
    return c
swag = paramcont(1,[-2,2])
# %%



# %%
