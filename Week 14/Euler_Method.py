import numpy as np
import matplotlib.pyplot as plt
import math

f = lambda t, x:x
"Defining the ODE"

"Creating the individual step function for Eulers method"
def euler_step(xn,t,h):
    xn1 = xn + h*f(t,xn)
    "Single step Euler function"
    return xn1

"Creating the function for approximation using Eulers method"
def solve_toEU(x1,t1,t2,h):
    deltat_max = 1
    if h >= deltat_max:
        print('Step size too large')
        exit(solve_toEU)


    t = np.arange(t1,t2+h,h)
    x = np.array([x1])
    error = 0
    y = np.exp(t)

    for i in range(0,len(t)-1):
        xnplus1 = euler_step(x[i],t[i],h)
        x = np.append(x,xnplus1)

    for j in range(0,len(x)):
        err = y[j]-x[j]
        error += err
        print(error)





"Creating the individual step function for the 4th order Runge-Kutta method"
def rk_step(xn,t,h):
    k1 = f(t,xn)
    k2 = f(t+h/2,xn + k1/2)
    k3 = f(t+h/2,xn+h*(k2/2))
    k4 = f(t+h,xn+(h*k3))
    xn1 = xn + (h/6)*(k1 + 2*k2 + 2*k3 + k4)
    print(xn1)
    return xn1

"Creating the function for approximation using the 4th order Runge-Kutta method"
def solve_toRK(x1,t1,t2,h):
    deltat_max = 0.5
  
    if h >= deltat_max:
        print('Step size too large')
        exit(solve_toRK)

    t = np.arange(t1,t2+h,h)
    y = np.exp(t)
    x = np.array([x1])

    for i in range(0,len(t)-1):
        xnplus1 = rk_step(x[i],t[i],h)
        x = np.append(x,xnplus1)
    plt.plot(t,x,'b-',label='Runge Kutta method')
    plt.plot(t,y,'r-',label = 'Actual function')
    plt.legend()
    plt.title('Runge Kutta method approximation for an ODE')
    plt.xlabel('t')
    plt.ylabel('f(t)')
    plt.grid()
    plt.show()


def solve_to(x1,t1,t2,h):
    method = input("Which approximation method would you like to use? Please input either EU or RK EU")
    if method == "RK":
        solve_toRK(x1,t1,t2,h)
    elif method == "EU":
        solve_toEU(x1,t1,t2,h)
    else:
        print("Please provide a correct input")
        exit(solve_to)
    
solve_toEU(1,0,1,0.7)