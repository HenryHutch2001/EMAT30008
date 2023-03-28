#TESTING
import numpy as np
import matplotlib.pyplot as plt
from scipy import root


def PDE(x):
    return 0

a = 1
b = 2
N = 20
GridSpace = np.linspace(a,b,N-1)
dx = (b-a)/N


 
def FiniteSolver(PDE,a,b,dx):
    x0 = a
    xn = b
    xi = np.zeros(N-1)
    
