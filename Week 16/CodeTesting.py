# %%
import scipy
from scipy.optimize import root 
import matplotlib.pyplot as plt
import numpy as np
from math import nan
from My_Functions import shooting,solve_to
# %%

def ode(t,y):
    sigma = -1
    beta = 1
    u1 = y[0]
    u2 = y[1]
    du1_dt = np.float64(beta*u1 - u2 + sigma*u1*(u1**2 + u2**2))
    du2_dt = np.float64(u1 + beta*u2 +sigma*u2*(u1**2 + u2**2))
    return [du1_dt, du2_dt]

shooting([1,1,20],ode) #Our shooting method determines the limit cycle of the ode above to have initial conditions x0 = -1.00055095e+00  y0 = 1.10402831e-03
# and to have a period of T = 1.88573782e+01 seconds
t,x = solve_to(ode,[1,1],0,10,0.1)
plt.plot(x[:,0],x[:,1])
plt.xlabel('x')
plt.ylabel('y')
plt.title('Phase portrait of the ODE')
plt.show()

# %%
