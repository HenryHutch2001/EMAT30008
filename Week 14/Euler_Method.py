import numpy as np
import matplotlib.pyplot as plt

f = lambda t, x:x

def euler_step(xn,t,h):
    xn1 = xn + h*f(t,xn)
    return xn1

def solve_to(x1,t1,t2,h):
    deltat_max = 0.5
    if h >= deltat_max:
        print('Step size too large')
        exit(solve_to)
    t = np.arange(t1,t2+h,h)
    x = np.array([x1])
    for i in range(0,len(t)-1):
        xnplus1 = euler_step(x[i],t[i],h)
        x = np.append(x,xnplus1)
    plt.plot(t,x,'bo--')
    plt.title('Eulers method approximation for an ODE')
    plt.xlabel('t')
    plt.ylabel('f(t)')
    plt.grid()
    plt.show()

solve_to(1,1,5,0.4)
