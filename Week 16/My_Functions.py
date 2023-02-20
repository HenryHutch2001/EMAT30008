import numpy as np
import matplotlib.pyplot as plt
from sys import exit

f = lambda t, x:x

def exact_solution(t):
    return np.exp(t) 
"Creating the individual step function for Eulers method"
def euler_step(xn,t,h):
    x = xn + h*f(t,xn)
    "Single step Euler function"
    return x
"Creating the function for approximation using Eulers method"
def solve_toEU(x0,t1,t2,h):
   deltat_max = 1
   if h >= deltat_max:
    print("Step size too large for accurate approximation")
    exit(solve_toEU)
   t = np.arange(t1,t2+h,h)
   x = np.zeros_like(t)
   x[0] = x0
   for i in range(1,len(t)):
        x[i] = euler_step(x[i-1],t[i-1],h)
   return t,x   
"Creating the individual step function for RK4 method"
def rk_step(xn,t,h):
    k1 = f(t,xn)
    k2 = f(t+h/2,xn + h*k1/2)
    k3 = f(t+h/2,xn+h*(k2/2))
    k4 = f(t+h,xn+(h*k3))
    xn1 = xn + (h/6)*(k1 + 2*k2 + 2*k3 + k4)
    return xn1
"Creating the function for approximation using RK4 method"
def solve_toRK(x0,t1,t2,h):
   deltat_max = 1
   if h >= deltat_max:
    print("Step size too large for accurate approximation")
    exit(solve_toRK)
   t = np.arange(t1,t2+h,h)
   x = np.zeros_like(t)
   x[0] = x0
   for i in range(1,len(t)):
        x[i] = rk_step(x[i-1],t[i-1],h)
   return t,x   

def solve_to(x0,t1,t2,h):
    deltat_max = 1
    if h >= deltat_max:
        print("Step size too large for accurate approximation")
        exit(solve_to)
    method = input("Which approximation method would you like to use? Please enter either Euler or Runge-Kutta ").lower()
    if method == ("Euler").lower():
        solve_toEU(x0,t1,t2,h)
    elif method == ("Runge-Kutta").lower():
        solve_toRK(x0,t1,t2,h)
    else:
        print("Please provide a correct input")
        exit(solve_to)