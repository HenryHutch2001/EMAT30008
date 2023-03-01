# %% Importing packages 
import scipy
from scipy.optimize import root 
import matplotlib.pyplot as plt
import numpy as np
from math import nan
from My_Functions import shooting,solve_to
import unittest
# %% Defining the ODE
def ode(t,y):
    sigma = -1
    beta = 1
    u1 = y[0]
    u2 = y[1]
    du1_dt = beta*u1 - u2 + sigma*u1*(u1**2 + u2**2)
    du2_dt = u1 + beta*u2 + sigma*u2*(u1**2 + u2**2)
    return [du1_dt, du2_dt]

# %%
x0 = [1,1,20]
shooting([1,1,20],ode) #Our shooting method determines the limit cycle of the ode above to have initial conditions x0 = -1.00055095e+00  y0 = 1.10402831e-03

""" # and to have a period of T = 1.88573782e+01 seconds
t,x = solve_to(ode,[1,1],0,10,0.1)
plt.plot(x[:,0],x[:,1])
plt.xlabel('x')
plt.ylabel('y')
plt.title('Phase portrait of the ODE')
plt.show()
 """
# %%
def u1(t, beta, theta):
    return np.sqrt(beta) * np.cos(t + theta)

def u2(t, beta, theta):
    return np.sqrt(beta) * np.sin(t + theta)

u2(1,1,2)
#%%
class TestApproximateLimitCycle(unittest.TestCase):
    def test_approximate_limit_cycle(self):
        beta = 1
        theta = np.pi *2
        tolerance = 1e-6
        approx_lc = shooting(x0,ode)
        # Compare approximate limit cycle with explicit solution
        for t in np.linspace(0, 2 * np.pi, 100):
            expected_u1 = u1(t, beta, theta)
            expected_u2 = u2(t, beta, theta)
            approx_u1 = approx_lc[0]
            approx_u2 = approx_lc[1]
            self.assertAlmostEqual(approx_u1, expected_u1, delta=tolerance)
            self.assertAlmostEqual(approx_u2, expected_u2, delta=tolerance)

if __name__ == '__main__':
    unittest.main()

